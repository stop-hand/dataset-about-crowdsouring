import sys
import numpy as np

answers_file = sys.argv[1]
ground_truth_file = sys.argv[2]

def argmax(a):
	max_ixs = []
	max_val = -10^10
	for i in xrange(len(a)):
		#print a[i], max_ixs
		if a[i] > max_val:
			max_ixs = [i]
			max_val = a[i]
		elif a[i] == max_val:
			max_ixs.append(i)
	return max_ixs[int(np.random.rand()*len(max_ixs))]

ground_truth = []
f_ground_truth = open(ground_truth_file)
classes = []
for line in f_ground_truth:
	c = int(line)
	ground_truth.append(c)
	if c not in classes:
		classes.append(c)
f_ground_truth.close()

mv_acc = 0.0
f_answers = open(answers_file)
fw = open(answers_file.replace("_sim.txt","_mv.txt"),"w")
i = 0
for line in f_answers:
	true = ground_truth[i]
	i += 1

	split = line.split(" ")
	votes = np.zeros(len(classes))
	for vote in split:
		vote = int(vote)
		if vote != -1:
			votes[vote] += 1
	#mv = np.argmax(votes)
	mv = argmax(votes)
	if mv == true:
		mv_acc += 1
	fw.write(str(mv) + "\n")
f_answers.close()
fw.close()
mv_acc /= i
print "majority voting accuracy: ", mv_acc
