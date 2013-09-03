import multiprocessing
def cv_loop_multi(args):
   f, X, y, model, N = args
return (cv_loop(X, y, model, N), f)

[...]

# Greedy feature selection loop
pool = multiprocessing.Pool(4)
while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
   scores = []
   args = []
   for f in range(len(Xts)):
      if f not in good_features:
         feats = list(good_features) + [f]
         Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
         args.append([f, Xt, y, model, N])
         #score = cv_loop(Xt, y, model, N)
         #scores.append((score, f))
         #print "Feature: %i Mean AUC: %f" % (f, score)
   cycle_score = pool.map(cv_loop_multi, args)
   for c in cycle_score:
      scores.append(c)

   good_features.add(sorted(scores)[-1][1])
   score_hist.append(sorted(scores)[-1])
   print "Current features: %s" % sorted(list(good_features))
   print "Best score was: %f" % (sorted(scores)[-1][1])

# terminate spawned processes
pool.terminate()