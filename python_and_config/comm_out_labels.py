import os

image_files = []
os.chdir(os.path.join("data", "custom", "labels"))

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".txt"):
#        txt_file= open(filename,"r")
#        new_txt_file= filename + "2"
#        line= txt_file.readline()
#        opts= [elem.strip() for elem in line.split(",")]
#        outfile= open(new_txt_file,'w')
#        print("outfile:",outfile)
#        outfile.writelines("%s %s %s %s %s\n" % (opts[0],opts[1],opts[2],opts[3],opts[4]))
#        outfile.close()
##        filepath = os.path.join(root, file)
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
#            print('%s read' % filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text.replace(',', ' '))
#            print('%s updated' % filename)

