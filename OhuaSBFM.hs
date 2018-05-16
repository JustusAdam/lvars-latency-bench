module Main where

import LatencyRunner
import OhuaSBFMBase
import Control.Monad.Stream.Chan

main = do
  makeMain (start_traverse runChanM) "sfbm-chans"
