from shlex  import split
import subprocess

bashCommand = "pytest pygyro --cov=pygyro --cov-report=term -k='not long'"
try:
    process = subprocess.run(split(bashCommand), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except:
    print(out)
    print(err)
    raise

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
