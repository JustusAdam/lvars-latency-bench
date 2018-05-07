{-# LANGUAGE BangPatterns, OverloadedStrings #-}
{-# LANGUAGE DoAndIfThenElse #-}

module Main where

import           Control.Concurrent (getNumCapabilities)
import           Control.Exception (evaluate)
import qualified Data.Set as Set
import qualified Data.IntSet as IS
import qualified Data.Vector as V

import           Data.Time.Clock (getCurrentTime, diffUTCTime)
import qualified Control.Parallel.Strategies as Strat
import System.IO.Unsafe

import LatencyRunner

bf_pure :: Int             -- iteration counter
           -> Graph2       -- graph
           -> IS.IntSet    -- set of "seen" node labels, initially size 0
           -> IS.IntSet    -- set of "new" node labels, initially size 1
           -> WorkFn       -- function to be applied to each node
           -> IS.IntSet
bf_pure 0 _ seen_rank new_rank _ = do
  -- when verbose $ prnt $ "bf_pure finished! seen/new size: "
  --   ++ show (IS.size seen_rank, IS.size new_rank)
  (IS.union seen_rank new_rank)

bf_pure k !g  !seen_rank !new_rank !f = do 
  -- when verbose $ prnt  $"bf_traverse call... "
  --   ++ show k ++ " seen/new size "
  --   ++ show (IS.size seen_rank, IS.size new_rank)
  if IS.null new_rank
  then seen_rank
  else do
    -- Add new_rank stuff to the "seen" list
    let seen_rank' = IS.union seen_rank new_rank
-- TODO: parMap version
--        allNbr     = IS.fold IS.union                      
        allNbr'    = IS.fold (\i acc -> IS.union (g V.! i) acc) 
                        IS.empty new_rank
        new_rank'  = IS.difference allNbr' seen_rank' 
    bf_pure (k-1) g seen_rank' new_rank' f


start_traverse :: Starter
start_traverse k !g startNode f = do
    ncap <- getNumCapabilities
    lock <- newLock
    putStrLn $ " * Running on " ++ show ncap ++ " parallel resources..."
    let set = bf_pure k g IS.empty (IS.singleton startNode) f
        l =
            Strat.withStrategy
                (Strat.parBuffer 16 Strat.rdeepseq)
                (map (unsafePerformIO . withLock lock . withTimeStamp f) (IS.toList set))
        set2 = Set.fromList $ map fst l
        size = Set.size set2
    t0 <- getCurrentTime
    begin <- currentTimeMillis
    evaluate set
    t1 <- getCurrentTime
    putStrLn $
        " * Bf_pure result in WHNF (done with bf_pure?) " ++
        show (diffUTCTime t1 t0)
    putStr $ "  * Set size: " ++ show size ++ " "
    t2 <- getCurrentTime
    print (diffUTCTime t2 t1)
    evaluate set2
    let ls = Set.toList set2
    putStrLn $ " * first element of result: " ++ show (head ls)
    putStrLn $ " * first 10 elements of result: " ++ show (take 10 ls)
    t3 <- getCurrentTime
    putStrLn $
        " * Analyze function should be finished! " ++ show (diffUTCTime t3 t2)
    putStrLn $ " * Set sum: " ++ show (Set.fold (\(_, x) y -> x + y) 0 set2)
    pure $ begin : map snd l

main = makeMain start_traverse "strategies"
