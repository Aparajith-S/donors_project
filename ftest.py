import finding_donors
def ftest():
	xaccuracy=None
	xfscore=None
	xaccuracy,xfscore = train_test()
	if xaccuracy is not None:
		return 1
	else:
		return 0
