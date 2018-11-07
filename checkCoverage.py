
bashCommand = "pytest pygyro/initialisation --cov=pygyro --cov-report=term"
import subprocess
process = subprocess.run(bashCommand.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

out = str(process.stdout)
out = out[2:-1]
out = out.replace("\\n","\n")

err = str(process.stderr)
err = err[2:-1]
err = err.replace("\\n","\n")

print(out)

print(err)

if (process.returncode==0):
    i = out.find("TOTAL")
    totline = out[i:]
    totline = totline[:totline.find("\n")]
    pc=float(totline[totline.rfind(" "):totline.rfind("%")])
    assert(pc>=75)
