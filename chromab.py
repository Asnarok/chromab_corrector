import sys
import corrector
args = sys.argv


argument = ""
for a in args:
    argument = argument + " " + a

isolated_args = argument.split("--")
isolated_args = isolated_args[1:]
args = []
for e in isolated_args:
    splitted = e.split(" ")
    print(e)
    #applying type conversions for proper compatibility
    if splitted[0] != "file":
        splitted[1] = float(splitted[1])
        if splitted[1] - int(splitted[1]) == 0:
            splitted[1] = int(splitted[1])
    args.append(splitted)

for e in args:
    corrector.settings[e[0]] = e[1]
print(corrector.settings)
corrector.perform_correction()