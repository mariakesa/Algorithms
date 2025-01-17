let x=1234;
let y=5678;

function karatsuba(x,y){
    let n_digits=Math.max(x.toString().length,y.toString().length);
    if(n_digits<=1){
        return x*y;
    }
    let m=Math.floor(n_digits/2);
    let a=Math.floor(x/10**m);
    let b=x%10**m;
    let c=Math.floor(y/10**m);
    let d=y%10**m;
    let partial1=karatsuba(a,c);
    let partial2=karatsuba(b,d);
    let partial3=karatsuba(a+b,c+d);
    return partial1*10**(2*m)+(partial3-partial1-partial2)*10**m+partial2;
}

console.log(karatsuba(x,y));
console.log(x*y);
