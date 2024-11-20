#!/bin/bash

echo "Starting test of the assignment"

export OLD_ASSIGNMENT_TO_RUN="cv03"
export ASSIGNMENT_TO_RUN="cv04"
export PYTHONPATH=`pwd`

prepare_cv01 ()
{
  echo "Preparing unittests for CV1"
  echo "... nothing to prepare ..."
}

prepare_cv02 ()
{
  echo "Preparing unittests for CV2"
  rm cv02/consts.py
  cp ~/consts.py cv02/consts.py
  echo "... cv02 prepared ..."
}

prepare_cv03 ()
{
  echo "Preparing unittests for CV3"
  rm cv02/consts.py
  cp ~/consts.py cv02/consts.py
  echo "... cv03 prepared ..."
}

prepare_cv04 ()
{
  echo "Preparing unittests for CV4"
  echo "... cv04 prepared ..."
}


prepare_cv05 ()
{
  echo "Preparing unittests for CV5"
  echo "... cv05 prepared ..."
}

eval prepare_cv01
eval prepare_cv02
eval prepare_cv03
eval prepare_cv04
python3 -m unittest test_cv_01.py
python3 -m unittest test_cv_02.py
python3 -m unittest test_cv_03.py
python3 -m unittest test_cv_04.py
# prepare the tests as necessary
# eval prepare_$OLD_ASSIGNMENT_TO_RUN
# eval prepare_$ASSIGNMENT_TO_RUN

# # run the tests
# # python3 -m unittest test_`echo $OLD_ASSIGNMENT_TO_RUN`.py
# python3 -m unittest test_`echo $ASSIGNMENT_TO_RUN`.py
exit $?
