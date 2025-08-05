print ("select an operation:")
print ("1. add")
print ("2. subtract")
print ("3. multiply")
print ("4. divide")
operation = input ()
if operation == "1":
   NUM1 = float(input ("Enter first number: "))
   NUM2 = float(input ("Enter second number: "))
   print ("The sum is: ",NUM1 + NUM2)
elif operation == "2":
   NUM1 = float(input ("Enter first number: "))
   NUM2 = float(input ("Enter second number: "))
   print ("The difference is: ",NUM1 - NUM2)
elif operation == "3":
   NUM1 = float(input ("Enter first number: "))
   NUM2 = float(input ("Enter second number: "))
   print ("The product is: ",NUM1 * NUM2)

elif operation == "4":
   NUM1 = float(input ("Enter first number: "))
   NUM2 = float(input ("Enter second number: "))
   print ("The quotient is: ",NUM1 / NUM2)
else :
  print ("invalid operation") 