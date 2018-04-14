# Kenan Sakarcan Homework 1 Airlines

airlines = """Aer Lingus,Aeroflot*,Aerolineas Argentinas,Aeromexico*,Air Canada,Air France,
Air India*,Air New Zealand*,Alaska Airlines*,Alitalia,All Nippon Airways,American*,
Austrian Airlines,Avianca,British Airways*,Cathay Pacific*,China Airlines,Condor,
COPA,Delta / Northwest*,Egyptair,El Al,Ethiopian Airlines,Finnair,Garuda Indonesia,
Gulf Air,Hawaiian Airlines,Iberia,Japan Airlines,Kenya Airways,KLM*,Korean Air,LAN Airlines,
Lufthansa*,Malaysia Airlines,Pakistan International,Philippine Airlines,Qantas*,
Royal Air Maroc,SAS*,Saudi Arabian,Singapore Airlines,South African,Southwest Airlines,
Sri Lankan / AirLanka,SWISS*,TACA,TAM,TAP - Air Portugal,Thai Airways,Turkish Airlines,
United / Continental*,US Airways / America West*,Vietnam Airlines,Virgin Atlantic,
Xiamen Airlines"""


""" PART 1 """
''' How many airlines are there '''
airlines_count = 1
for count in airlines:
        if count == ',':
            airlines_count += 1

#Note for Part 1, question 1
'''this result shows that there are 56 airlines
#set airline_count to plus one because the last airline "Xiamen" doesn't have
a comma'''
 
''' How many airlines start with the letter 'A' (only capital letters) '''

#convert airlines into a list
airlines = airlines.split(',')

airlines_withA = 0
for count in airlines:
    if count[0] == 'A':
        airlines_withA +=1
        print airlines_withA
        

#Note for part 1, question 2
'''Split airlines by comma separation if count, if the first position first 
letter is an A, than add to airlineswithA count, the result was 12 airlines


''' create a list that only includes airlines that start with the letter "A" '''

l1 = []
for item in airlines:
    if item[0] == ('A'):
        l1.append(item)
        print item

#Note, I'm not sure why this prints item as Xiamen Airlines to my variable exp
#I got 12 and the answer is 14....I'm not sure why
        

''' Create a list (of the same length) that contains 1 if there's a star and 0 if not.
Expected output: [0, 1, 0, ...] '''

airlines

#Value   #Condition                            #list to go through
l2= [1      if airline[-1] == '*'  else 0        for airline in airlines]



""" PART 2 """
''' Create a list of airline names (include all airlines, but not the asterisk).
Expected output: ['Aer Lingus', 'Aeroflot', 'Aerolineas Argentinas', ...] '''
airlines

airlines_wo_asterisk = [airline.replace('*',' ').replace('\n',' ') for airline 
in airlines]

#Note, I replaced the * with spaces and replaced the /ns with spaces, for in the 
airlines list



# number of available seat kilometers (ASKs), which is defined as the number
# of seats multiplied by the number of kilometers the airline flies
ask = '320906734,1197672318,385803648,596871813,1865253802,3004002661,869253552,\
710174817,965346773,698012498,1841234177,5228357340,358239823,396922563,\
3179760952,2582459303,813216487,417982610,550491507,6525658894,557699891,\
335448023,488560643,506464950,613356665,301379762,493877795,1173203126,\
1574217531,277414794,1874561773,1734522605,1001965891,3426529504,1039171244,\
348563137,413007158,1917428984,295705339,682971852,859673901,2376857805,\
651502442,3276525770,325582976,792601299,259373346,1509195646,619130754,\
1702802250,1946098294,7139291291,2455687887,625084918,1005248585,430462962'


''' Create a dictionary in which the key is the airline name (without the star)
and the value is the average number of incidents.


Expected output: {'Aer Lingus': 0.07, 'Aeroflot': 2.73, ...} '''

# I don't understand this question 

