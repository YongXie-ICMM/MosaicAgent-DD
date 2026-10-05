#!/bin/zsh
cd -- "${0:A:h}" || exit 1
python3 server.py "$@"
if (( $? != 0 )); then
  read "?启动未完成，请保留提示。按回车关闭。"
fi
