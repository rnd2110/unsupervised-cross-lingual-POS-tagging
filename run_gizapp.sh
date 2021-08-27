
mkdir workspace/$1-$2-$3-gfiles

rm -rf workspace/$1-$2-$3-gfiles/*

cp workspace/$1-$2-$3.txt workspace/$1-$2-$3-gfiles/$1-$2

cp workspace/$2-$1-$3.txt workspace/$1-$2-$3-gfiles/$2-$1

cd mkcls-v2

./mkcls -n10 -p../workspace/$1-$2-$3-gfiles/$1-$2 -V../workspace/$1-$2-$3-gfiles/$1-$2.vcb.classes

./mkcls -n10 -p../workspace/$1-$2-$3-gfiles/$2-$1 -V../workspace/$1-$2-$3-gfiles/$2-$1.vcb.classes

cd ../GIZA++

./plain2snt.out ../workspace/$1-$2-$3-gfiles/$1-$2 ../workspace/$1-$2-$3-gfiles/$2-$1

./snt2cooc.out ../workspace/$1-$2-$3-gfiles/$1-$2.vcb ../workspace/$1-$2-$3-gfiles/$2-$1.vcb ../workspace/$1-$2-$3-gfiles/$1-$2_$2-$1.snt > ../workspace/$1-$2-$3-gfiles/$1-$2.cooc

./GIZA++ -S ../workspace/$1-$2-$3-gfiles/$1-$2.vcb -T ../workspace/$1-$2-$3-gfiles/$2-$1.vcb -C ../workspace/$1-$2-$3-gfiles/$1-$2_$2-$1.snt -CoocurrenceFile ../workspace/$1-$2-$3-gfiles/$1-$2.cooc -o g -outputpath ../workspace/$1-$2-$3-gfiles

./GIZA++ ../workspace/$1-$2-$3.giza.config

mv $1_$2-$3.dict.A3.final ../workspace/$1-$2-$3-gfiles/$1-$2.alignments



