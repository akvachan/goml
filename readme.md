# goml - ML with Go

## Optimizations

- [ ] SIMD Support

## Algorithms

- [x] Perceptron
- [ ] Logistic Regression

## Misc

### Profiling

```
cd bin/ && go build ../main.go && ./main -cpuprofile=".prof" && go tool pprof main .prof && cd ..
```
