#!/bin/bash

echo "Starting test of the assignment"

export OLD_ASSIGNMENT_TO_RUN="cv01"
export ASSIGNMENT_TO_RUN="cv02"
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
  cp ~/fasttext_cc.cs.300.vec fasttext_cc.cs.300.vec
  echo "... cv02 prepared ..."
}

prepare_cv03 ()
{
  echo "Preparing unittests for CV3"
  rm cv02/consts.py
  cp ~/consts.py cv02/consts.py
  cp ~/fasttext_cc.cs.300.vec fasttext_cc.cs.300.vec
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

# prepare the tests as necessary
eval prepare_$OLD_ASSIGNMENT_TO_RUN
eval prepare_$ASSIGNMENT_TO_RUN

# run the tests
# python3 -m unittest test_`echo $OLD_ASSIGNMENT_TO_RUN`.py
python3 -m unittest test_`echo $ASSIGNMENT_TO_RUN`.py
exit $?
