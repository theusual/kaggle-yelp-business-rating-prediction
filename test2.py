dfAll[1] = dfAll[1].reset_index(drop=True)
j=0
for row in dfAll[1].ix[:,['bus_categories']].values:
    for list in row:
        for i in list:
            if i == 'Mortgage Brokers':
                print dfAll[1]['bus_review_count'][j]
    j+=1