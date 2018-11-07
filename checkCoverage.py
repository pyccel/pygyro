

search = open("current_coverage.txt","r")

for line in search:
    if ("TOTAL" in line):
        i=line.rfind(" ")
        j=line.rfind("%")
        pc=float(line[i:j])
        assert(pc>=75)

search.close()
