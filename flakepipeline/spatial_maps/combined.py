#!/usr/bin/env python3
"""Combine 1L/2L/TL in each spatial tile, with area-weighted margin profiles."""
from __future__ import annotations

import csv
from pathlib import Path

from . import heatmaps as source
from .heatmaps import format_pct
from .stats import require
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Wedge, Circle
import numpy as np

COLORS = ("#008000", "#800000", "#808000")
NAMES = ("1L", "2L", "TL")


def fractions(counts):
    denominator = np.sum(counts, axis=-1, keepdims=True)
    return np.divide(counts*100., denominator, out=np.full(counts.shape, np.nan), where=denominator > 0)


def prepare(result):
    columns, rows = len(result["xedges"])-1, len(result["yedges"])-1
    table, whole = result["table"], result["global"]
    counts = np.array([[t[f"{k}_px"] for k in NAMES] for t in table], dtype=np.int64).reshape(rows, columns, 3)
    row_counts, col_counts = counts.sum(axis=1), counts.sum(axis=0)
    total = np.array([whole["counts_px"][k] for k in NAMES])
    require(np.array_equal(row_counts.sum(axis=0),total), "Row profile counts not conserved")
    require(np.array_equal(col_counts.sum(axis=0),total), "Column profile counts not conserved")
    return dict(result, dose=result["sulfur_mg"], xs=result["xedges"], ys=result["yedges"], whole=whole,
                row_counts=row_counts, col_counts=col_counts,
                row_pct=fractions(row_counts), col_pct=fractions(col_counts))


def map_frame(ax, data):
    xs, ys = data["xs"], data["ys"]
    ax.set(xlim=(0, data["width"]), ylim=(data["height"], 0), aspect="equal")
    ax.set_xticks((xs[:-1]+xs[1:])/2, [str(i+1) for i in range(len(xs)-1)])
    ax.set_yticks((ys[:-1]+ys[1:])/2, [str(i+1) for i in range(len(ys)-1)])
    ax.set_xlabel("Column (left → right)", fontsize=10)
    ax.set_ylabel("Row (top → bottom)", fontsize=10)
    ax.tick_params(length=0, labelsize=9, pad=5)
    for spine in ax.spines.values():spine.set(color="#bfc5c8", linewidth=.65)


def spatial(ax, data, style):
    for tile in data["table"]:
        x, y = tile["x0_px"], tile["y0_px"]
        w, h = tile["x1_px"]-x, tile["y1_px"]-y
        empty = not tile["retained_canvas_px"]
        partial = tile["excluded_px"] + tile["non_scan_px"] > 0 and not empty
        ax.add_patch(Rectangle((x,y),w,h, facecolor="#eceff1" if empty or partial else "#fbfcfc",
                               edgecolor="#d7dcdf", linewidth=.6,
                               hatch="////" if empty else None, zorder=0))
        if partial:
            ax.text(x+w*.92,y+h*.12,"†",ha="center",va="center",fontsize=9,color="#515a60")
        if empty:
            ax.text(x+w/2,y+h/2,"—",color="#8b969c",ha="center",va="center",fontsize=12)
            continue
        if not tile["crystal_px"]:
            ax.text(x+w/2,y+h/2,"No crystal",color="#6b7479",ha="center",va="center",fontsize=6)
            continue
        values = np.array([tile[f"{k}_px"] for k in NAMES])/tile["crystal_px"]
        if style == "pies":
            # Angles remain complete, including in partial tiles. Exclusions
            # change the underlying denominator, never clip sectors visually.
            angle = 90.
            radius = min(w,h)*.37
            for value, color in zip(values,COLORS):
                ax.add_patch(Wedge((x+w/2,y+h/2),radius,angle,angle+value*360,
                                  facecolor=color,edgecolor="white",linewidth=.6))
                angle += value*360
            ax.add_patch(Circle((x+w/2,y+h/2),radius,fill=False,edgecolor="#475159",linewidth=.35))
        else:
            left = x+w*.075
            for value, color in zip(values,COLORS):
                ax.add_patch(Rectangle((left,y+h*.24),w*.85*value,h*.52,
                                      facecolor=color,edgecolor="white",linewidth=.55))
                left += w*.85*value
    map_frame(ax,data)


def row_profile(ax, data):
    ys = data["ys"]; centers=(ys[:-1]+ys[1:])/2; heights=np.diff(ys)*.7
    left=np.zeros(len(centers))
    for index,color in enumerate(COLORS):
        values=data["row_pct"][:,index]
        ax.barh(centers,values,height=heights,left=left,color=color,edgecolor="white",linewidth=.5)
        for y,value,x in zip(centers,values,left):
            if value >= 9:ax.text(x+value/2,y,f"{value:.0f}",ha="center",va="center",color="white",fontsize=8)
        left+=np.nan_to_num(values)
    ax.set(xlim=(0,100),ylim=(data["height"],0),xlabel="Layer composition (%)")
    ax.set_xticks([0,25,50,75,100]);ax.set_yticks([])
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title("Composition by row",fontsize=11,pad=12)


def column_profile(ax,data):
    xs=data["xs"];centers=(xs[:-1]+xs[1:])/2;widths=np.diff(xs)*.7;bottom=np.zeros(len(centers))
    for index,color in enumerate(COLORS):
        values=data["col_pct"][:,index]
        ax.bar(centers,values,width=widths,bottom=bottom,color=color,edgecolor="white",linewidth=.5)
        for x,value,y in zip(centers,values,bottom):
            if value>=11:ax.text(x,y+value/2,f"{value:.0f}",ha="center",va="center",color="white",fontsize=8)
        bottom+=np.nan_to_num(values)
    ax.set(xlim=(0,data["width"]),ylim=(0,100),ylabel="Composition (%)",xlabel="Column")
    ax.set_xticks(centers,[str(i+1) for i in range(len(centers))]);ax.set_yticks([0,50,100])
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title("Composition by column",fontsize=11,pad=9)


def draw(data,out,style):
    fig=plt.figure(figsize=(8.6,10.5),facecolor="white")
    height=min(.625, .46*8.6/10.5*data["height"]/data["width"]);width=height*10.5/8.6*data["width"]/data["height"]
    mapax=fig.add_axes([.10,.275,width,height])
    rowax=fig.add_axes([.64,.275,.31,height])
    colax=fig.add_axes([.10,.075,width,.12])
    totalax=fig.add_axes([.64,.075,.31,.12]);totalax.axis("off")
    spatial(mapax,data,style);row_profile(rowax,data);column_profile(colax,data)
    mapax.set_title("Each tile contains 1L + 2L + TL",fontsize=11,pad=12)
    totalax.set_title("Whole-map composition",fontsize=11,pad=9)
    for i,(layer,color) in enumerate(zip(NAMES,COLORS)):
        y=.84-i*.31
        totalax.add_patch(Rectangle((.08,y-.065),.11,.13,facecolor=color,transform=totalax.transAxes))
        totalax.text(.27,y,layer,va="center",fontsize=11,transform=totalax.transAxes)
        totalax.text(.93,y,format_pct(data['whole']['of_crystal_pct'][layer]),ha="right",va="center",fontsize=11,transform=totalax.transAxes)
    fig.suptitle(f"{data['panel']} · spatial layer composition",fontsize=15,y=.977)
    fig.legend([Patch(facecolor=c) for c in COLORS],NAMES,loc="upper center",bbox_to_anchor=(.5,.956),ncol=3,
               frameon=False,fontsize=11,handlelength=1.5,columnspacing=2.5)
    glyph="pies" if style=="pies" else "bars"
    fig.text(.5,.018,f"Equal-size {glyph}: fractions within local MoS₂ area. † Partly omitted tile; hatched: no retained scan support.\n"
             f"Margin profiles use summed pixel areas. TL = thicker layers. Grid: {len(data['xs'])-1} columns × {len(data['ys'])-1} rows.",ha="center",fontsize=8.4,color="#40484d")
    source.save(fig,out,f"{data['panel']}_combined_{style}")


def export_data(data,out):
    with (out/f"{data['panel']}_tiles_{len(data['xs'])-1}x{len(data['ys'])-1}.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(data["table"][0]));writer.writeheader();writer.writerows(data["table"])
    records=[]
    for direction,counts in [("row",data["row_counts"]),("column",data["col_counts"])]:
        p=fractions(counts)
        for i,(count,pct) in enumerate(zip(counts,p)):
            record={"direction":direction,"index_1based":i+1,"crystal_px":int(count.sum())}
            record.update({f"{k}_px":int(count[j]) for j,k in enumerate(NAMES)})
            record.update({f"{k}_pct":float(pct[j]) if np.isfinite(pct[j]) else None for j,k in enumerate(NAMES)})
            records.append(record)
    with (out/f"{data['panel']}_margin_profiles.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def overview(all_data,out,style):
    fig,axes=plt.subplots(1,len(all_data),figsize=(max(8,4*len(all_data)),8),squeeze=False)
    axes=axes.ravel()
    fig.subplots_adjust(left=.06,right=.99,bottom=.11,top=.86,wspace=.25)
    for ax,data in zip(axes,all_data):
        spatial(ax,data,style)
        ax.set_title(str(data["panel"]),fontsize=13,pad=10)
    fig.suptitle("Spatial layer composition · all three layers in every tile",fontsize=15,y=.98)
    fig.legend([Patch(facecolor=c) for c in COLORS],NAMES,loc="upper center",bbox_to_anchor=(.5,.946),ncol=3,frameon=False,fontsize=11)
    fig.supxlabel(f"{len(all_data[0]['xs'])-1} × {len(all_data[0]['ys'])-1} tiles per sample · 1L + 2L + TL = 100% of local MoS₂ area\n"
                  "† Partly omitted tile; hatched: no retained scan support. Tile symbol size does not encode crystal coverage.",fontsize=9)
    source.save(fig,out,f"all_samples_combined_{style}")
