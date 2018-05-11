{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DoAndIfThenElse, OverloadedLists, MultiWayIf, OverloadedStrings #-}

module Main where

import Monad.Generator as G
import LatencyRunner
import qualified Data.IntSet as IS
import qualified Data.Set as Set 
import Control.DeepSeq
import Monad.StreamsBasedFreeMonad
import Monad.StreamsBasedExplicitAPI
import Control.Monad.Trans.Class
import Control.Monad.State
import qualified Data.Vector as V
import Data.Time
import Data.Aeson
import qualified Data.ByteString.Lazy as BS
import System.CPUTime
import Data.Dynamic2
import Text.Printf


bf_generate :: Int -> Int -> Graph2 -> WorkFn -> () -> StateT () IO (Generator IO Int)
bf_generate k0 startNode g f () =
    pure $
    let gen !seen_rank !k !new_rank
            | k == 0 = finishTraverse
            | IS.null new_rank = finishTraverse
            | otherwise = do
                let seen_rank' = IS.union seen_rank new_rank
                    allNbr' =
                        IS.fold
                            (\i acc -> IS.union (g V.! i) acc)
                            IS.empty
                            new_rank
                    new_rank' = IS.difference allNbr' seen_rank'
                    ls = IS.toList new_rank'
                    res = map (snd . f) ls
                forceA res
                (foldableGenerator res `mappend`
                 gen seen_rank' (pred k) new_rank')
          where
            finishTraverse =
              finish
          
     in gen mempty k0 [startNode]



forceA :: (Applicative m, NFData a) => a -> m a
forceA a = a `deepseq` pure a

{-# INLINE withUnitState #-}
withUnitState :: StateT () IO a -> StateT () IO a
withUnitState = id

start_traverse :: Starter
start_traverse k g startNode f = do
    algo <-
        createAlgo $ do
            unit <- sfConst' ()
            nodeStream <- liftWithIndex 0 (bf_generate k startNode g f) unit
            processedStream <-
                smapGen
                    (liftWithIndex 1 $ withUnitState . withTimeStamp f)
                    nodeStream
            liftWithIndex
                2
                (forceA . map snd <=< withUnitState . liftIO . G.toList)
                processedStream
    begin <- currentTimeMillis
    stamps <-
        runAlgo algo $
        let unit = toDyn ()
         in replicate 3 unit
    pure $ begin : stamps
    --putStrLn $ "  * Set size: " ++ show (Set.size set)
    --putStrLn $ "  * Set sum: " ++ show (Set.foldr (\(x,_) y -> x+y) 0 set)

    


main = do
  makeMain start_traverse "sfbm"
