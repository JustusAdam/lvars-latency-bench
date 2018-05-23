{-# LANGUAGE BangPatterns, OverloadedStrings #-}
{-# LANGUAGE DoAndIfThenElse, LambdaCase #-}

module LatencyRunner where

-- This is used with three different benchmark implementations:
--   PURE        -- use LVarTracePure
--   STRATEGIES  -- use Strategies, no LVars
--   PAR         -- use Par, no LVars
--
-- Run-time options:
--   Wrk = work to do per vertex
--   depthK = max hops of the connected component to explore
--   (OR N = target vertices to visit (will overshoot))

import           Control.Exception (evaluate)
import           Control.Monad (forM_, when)
import           Data.Word
import           Data.IORef
import           Data.List as L
import           Data.List.Split (chunksOf)
import qualified Data.IntSet as IS
import qualified Data.ByteString.Lazy.Char8 as B
import           Data.Time.Clock (getCurrentTime, diffUTCTime)
import           Text.Printf (printf)
import           System.Mem (performGC)
import           System.IO.Unsafe (unsafePerformIO)
import           System.Environment (getEnvironment,getArgs)
import           System.CPUTime.Rdtsc (rdtsc)
import           System.CPUTime  (getCPUTime)
import Data.Time.Format
import Data.Time.Clock.POSIX
import Data.Time
import Control.DeepSeq
import Control.Monad.IO.Class
import Control.Concurrent.MVar
import Control.Exception
import Data.Aeson as JSON
import qualified Data.ByteString.Lazy as BL

-- For representing graphs
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV

-- Vector representation of graphs: the list (or set) at index k is
-- node k's neighbors.
type Graph = V.Vector [Node]
type Graph2 = V.Vector IS.IntSet

type Node = Int

-- Optimized version:
mkGraphFromFile :: String -> IO Graph
mkGraphFromFile file = do
    inStr <- B.readFile file
      -- Returns a list of edges:
    let loop1 [] = []
        loop1 (b1:b2:rst) = do
            case (B.readInt b1, B.readInt b2) of
                (Just (src, _), Just (dst, _)) -> (src, dst) : loop1 rst
                _ ->
                    error $
                    "Failed parse of bytestrings: " ++ show (B.unwords [b1, b2])
        loop1 _ = error "Odd number of integers in graph file!"
        edges =
            case B.words inStr of
                ("EdgeArray":rst) -> loop1 rst
        mx = foldl' (\mx (s, d) -> mx `max` s `max` d) 0 edges
    mg <- MV.replicate (mx + 1) []
    forM_ edges $ \(src, dst)
    -- Interpret this as a DIRECTED graph:
     -> do
        ls <- MV.read mg src
        MV.write mg src (dst : ls)
    V.freeze mg

-- Neighbors of a node with a given label
nbrs :: Graph -> Int -> [Int]
nbrs g lbl = g V.! lbl

-- For debugging
printGraph :: Graph -> IO ()
printGraph g = do
    let ls = V.toList g
    putStrLn (show ls)
    return ()

-- Iterates the sin function n times on its input and returns the sum
-- of all iterations.
sin_iter :: Word64 -> Float -> Float
sin_iter 0  x = x
sin_iter n !x = sin_iter (n - 1) (x + sin x)

type WorkRet = (Float, Node)
type WorkFn = (Node -> WorkRet)

theEnv :: [(String,String)]
theEnv = unsafePerformIO getEnvironment

checkEnv :: Read a => String -> a -> a
checkEnv v def =
  case lookup v theEnv of
    Just "" -> def
    Just s  -> read s
    Nothing -> def

verbose :: Bool
verbose = checkEnv "VERBOSE" False

dbg :: Bool
-- dbg = checkEnv "DEBUG" False
dbg = False -- Let it inline, DCE.

type Starter = Int       -- iteration counter
               -> Graph2 -- graph
               -> Int    -- start node
               -> WorkFn
               -> WorkFn -- function to be applied to each node
               -> IO [Integer]

currentTimeMillis :: IO Integer
currentTimeMillis = round . (* 1000) <$> getPOSIXTime

(<&>) = flip fmap

{-# INLINE makeMain #-}
makeMain :: Starter -> String -> IO ()
makeMain start_traverse ty = do
    let graphFile_ :: String
        graphFile_ = "/tmp/grid_125000"
    let k_ :: Int
        k_ = 25 -- Number of hops to explore
    let w_ :: Word64
        w_ = 20000 -- Amount of work (iterations of sin)
  -- LK: this way of writing the type annotations is the only way I
  -- can get emacs to not think this is a parse error! :(
    (graphFile, depthK, wrk, cwrk) <-
        getArgs <&> \case
            [] -> (graphFile_, k_, w_, w_)
            [graphFiles] -> (graphFiles, k_, w_, w_)
            [graphFiles, ks] -> (graphFiles, read ks, w_, w_)
            [graphFiles, ks, ws] -> (graphFiles, read ks, read ws :: Word64, w_)
            [graphFiles, ks, ws, cw] ->
                (graphFiles, read ks, read ws :: Word64, read cw)
    gr <- mkGraphFromFile graphFile
    let startNode = 0
        gr2 = V.map IS.fromList gr
    evaluate (gr2 V.! 0)
    let graphThunk :: WorkFn -> WorkFn -> IO [Integer]
        graphThunk fn0 fn1 = start_traverse depthK gr2 0 fn0 fn1
  -- Takes a node ID (which is just an int) and returns it paired with
  -- a floating-point number that's the value of iterating the sin
  -- function wrk times on that node ID.
    let sin_iter_count :: WorkFn
        sin_iter_count x = (sin_iter wrk $ fromIntegral x, x)
    let consumer_waiter :: WorkFn
        consumer_waiter n = unsafePerformIO $ wait_sins (fromIntegral cwrk) n
        producer_waiter :: WorkFn
        producer_waiter = unsafePerformIO . wait_sins (fromIntegral wrk)
    when verbose $
        printf
            "* Beginning benchmark with depthK=%d, wrk=%d and cwrk=%d \n"
            depthK
            wrk
            cwrk
    performGC
    t0 <- getCurrentTime
    ctime <- currentTimeMillis
    res <- graphThunk producer_waiter consumer_waiter
    ftime <- currentTimeMillis
    t1 <- getCurrentTime
    when verbose $ printf "SELFTIMED %d\n" (show (diffUTCTime t1 t0))
    BL.writeFile ("results-" ++ ty ++ "-" ++ tToStr t1) $
        encode $
        object
            [ "start" .= ctime
            --, "arrivals" .= res
            , "finish" .= ftime]
  where
    tToStr = formatTime defaultTimeLocale "%s"

------------------------------------------------------------------------------------------

-- Wait for a certain number of milleseconds.
wait_microsecs :: Word64 -> Node -> IO WorkRet
wait_microsecs clocks n = do
  myT <- rdtsc
  let loop !n = do
        now <- rdtsc
        if now - myT >= clocks
        then return n
        else loop (n+1)
  cnt <- loop 0
  return (fromIntegral cnt, n)


-- Measure clock frequency, spinning rather than sleeping to try to
-- stay on the same core.
measureFreq :: IO Word64 -- What units is this in? -- LK
measureFreq = do
    let millisecond = 1000 * 1000 * 1000 -- picoseconds are annoying
      -- Measure for how long to be sure?
        measure = 1000 * millisecond
        scale :: Integer
        scale = 1000 * millisecond `quot` measure
    t1 <- rdtsc
    start <- getCPUTime
    let loop :: Word64 -> Word64 -> IO (Word64, Word64)
        loop !n !last = do
            t2 <- rdtsc
            when (t2 < last) $ putStrLn $ "COUNTERS WRAPPED " ++ show (last, t2)
            cput <- getCPUTime
            if (cput - start < measure)
                then loop (n + 1) t2
                else return (n, t2)
    (n, t2) <- loop 0 t1
    putStrLn $
        "  Approx getCPUTime calls per second: " ++
        commaint (scale * fromIntegral n)
    when (t2 < t1) $
        printf
            "WARNING: rdtsc not monotonically increasing, first %d, then %d on the same OS thread"
            t1
            t2
    return $ fromIntegral (fromIntegral scale * (t2 - t1))

wait_sins :: Word64 -> Node -> IO WorkRet
wait_sins num node = do
  myT <- rdtsc
  res <- evaluate (sin_iter num (2.222 + fromIntegral node))
  return (res, node)

-- Measure the cost of N Sin operations.
measureSin :: Word64 -> IO Word64
measureSin n = do
    t0 <- rdtsc
    res <- evaluate (sin_iter n 38.38)
    t1 <- rdtsc
    return $ t1 - t0

-- Readable large integer printing:
commaint :: (Show a, Integral a) => a -> String
commaint n = reverse $ concat $ intersperse "," $ chunksOf 3 $ reverse (show n)


{-# INLINE withTimeStamp #-}
withTimeStamp :: (NFData b, MonadIO m) => (a -> b) -> a -> m (b, Integer)
withTimeStamp f =
    \i -> do
        ts <- liftIO currentTimeMillis
        let res = f i
        res `deepseq` pure (res, ts)


type Lock = MVar ()

{-# INLINE newLock #-}
newLock :: MonadIO m => m Lock
newLock = liftIO $ newMVar ()

withLock :: MonadIO m => Lock -> m a -> m a
withLock l ac = do
    () <- liftIO $ takeMVar l
    res <- ac
    liftIO $ putMVar l ()
    pure res
