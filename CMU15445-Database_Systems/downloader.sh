echo "Download course slides..."

mkdir slides

cd slides

wget -c -r -np -A.pdf https://15445.courses.cs.cmu.edu/fall2020/slides/

cd ..

echo "Download course notes..."

mkdir notes

cd notes

wget -c -r -np -A.pdf https://15445.courses.cs.cmu.edu/fall2020/notes/

cd ..