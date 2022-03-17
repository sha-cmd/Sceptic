import itertools

from collections import defaultdict


class Metrics:

    @staticmethod
    def GetTopN(predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)


        for userID, bookID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(bookID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    @staticmethod
    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0
        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutBookID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for bookID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutBookID) == int(bookID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    @staticmethod
    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutBookID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for bookID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutBookID) == bookID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    @staticmethod
    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutBookID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for bookID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutBookID) == bookID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    @staticmethod
    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutBookID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for bookID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutBookID) == bookID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    @staticmethod
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for bookID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    @staticmethod
    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                book1 = pair[0][0]
                book2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(book1)
                innerID2 = simsAlgo.trainset.to_inner_iid(book2)
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n if not n == 0 else 0
        return 1-S

    @staticmethod
    def Novelty(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                bookID = rating[0]
                rank = rankings[bookID]
                total += rank
                n += 1
        return total / n if not n == 0 else 0
