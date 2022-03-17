if pip freeze > requirements.txt; then
  echo -e "\e[32;1mDone !\e[0m"
else
  echo -e "\e[31;1mAn error occured\e[0m"
fi

git add requirements.txt