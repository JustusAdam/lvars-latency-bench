module Main where

import LatencyRunner
import OhuaSBFMBase
import Control.Monad.Stream.Par

main = do
  makeMain (start_traverse runParIO) "sbfmpar"
