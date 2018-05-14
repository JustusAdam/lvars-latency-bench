{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DoAndIfThenElse #-}

module Main where

import           Control.Exception (evaluate)
import           Control.Monad (forM_, when)
import qualified Data.Set as Set
import qualified Data.IntSet as IS
import qualified Data.Vector as V
import           GHC.Conc (numCapabilities)

import           Control.LVish
import           Data.LVar.PureSet
import System.CPUTime
import           Debug.Trace (trace)
import Control.Monad.IO.Class
import Data.Time.Clock.POSIX
import Control.Concurrent.MVar
import Control.DeepSeq

import LatencyRunner
import Control.LVish.Unsafe

prnt :: String -> Par d s ()
prnt str = trace str $ return ()

-- An LVar-based version of bf_traverse.  As we traverse the graph,
-- the results of applying f to each node accumulate in an LVar, where
-- they are available to other computations, enabling pipelining.

bf_traverse :: Int             -- iteration counter
               -> Graph2       -- graph
               -> ISet s Node -- LVar
               -> IS.IntSet    -- set of "seen" node labels, initially size 0
               -> IS.IntSet    -- set of "new" node labels, initially size 100
               -> WorkFn
               -> Par d s (IS.IntSet)
bf_traverse 0 _ _ seen_rank new_rank _ = do
  when verbose $ prnt $ "bf_traverse finished! seen/new size: "
    ++ show (IS.size seen_rank, IS.size new_rank)
  return (IS.union seen_rank new_rank)

bf_traverse k !g !l_acc !seen_rank !new_rank f = do 
  when verbose $ prnt  $"bf_traverse call... "
    ++ show k ++ " seen/new size "
    ++ show (IS.size seen_rank, IS.size new_rank)
  -- Nothing in the new_rank set means nothing left to traverse.
  if IS.null new_rank
  then return seen_rank
  else do
    -- Add new_rank stuff to the "seen" list
    let seen_rank' = IS.union seen_rank new_rank
        allNbr'    = IS.fold (\i acc -> IS.union (g V.! i) acc) 
                        IS.empty new_rank
        new_rank'  = IS.difference allNbr' seen_rank'
    
    fork $ mapM_ (`insert` l_acc) (map (snd . f) $ IS.toList new_rank')
    bf_traverse (k-1) g l_acc seen_rank' new_rank' f

start_traverse :: Int       -- iteration counter
                  -> Graph2 -- graph
                  -> Int    -- start node
                  -> WorkFn -- function to be applied to each node
                  -> IO [Integer]
start_traverse k !g startNode f = do
  begin <- currentTimeMillis
  lock <- newLock
  runParIO $ do        
        prnt $ " * Running on " ++ show numCapabilities ++ " parallel resources..."
        
        l_acc <- newEmptySet
        l_res <- newEmptySet
        tsTracker <- newEmptySet
        pool <- newPool

        lock <- liftIO $ newMVar ()
        
        -- "manually" add startNode
        fork $ insert (f startNode) l_res
        -- pass in { startNode } as the initial "new" set
        --set <- bf_traverse k g l_acc IS.empty (IS.singleton startNode)
        
        forEachHP (Just pool) l_acc (\i -> withLock lock $ do 
                          
                          arrivalStamp <- liftIO currentTimeMillis
                          (res, ts) <- withTimeStamp f i
                          insert res l_res
                          insert (i, arrivalStamp) tsTracker
                          
                      )
        set <- bf_traverse k g l_acc IS.empty (IS.singleton startNode) f
        
        prnt $ " * Done with bf_traverse..."
        let size = IS.size set
        
        prnt$ " * Waiting on "++show size++" set results..."

        when dbg $ do 
          forM_ [0..size] $ \ s -> do
            prnt$ " ? Blocking on "++show s++" elements to be in the set..."
            waitSize s l_acc

        -- Waiting is required in any case for correctness, whether or
        -- not we consume the result
        waitSize (size) l_res -- Depends on a bunch of forked computations
        prnt$ " * Set results all available! (" ++ show size ++ ")"

        s <- freezeSet l_res
        quiesce pool
        ts <- map snd . Set.toList <$> freezeSet tsTracker
        liftIO (do evaluate s; return ())
        prnt $ " * Finished consumeSet:"
        prnt $ "  * Set size: " ++ show (Set.size s)
        prnt $ "  * Set sum: " ++ show (Set.fold (\(x,_) y -> x+y) 0 s)
        pure $ begin : ts


main = makeMain start_traverse "LVars"

