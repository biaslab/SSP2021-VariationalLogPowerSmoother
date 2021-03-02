using WAV, DSP

function load_data(file::String, duration::Int64; noise_db::Int64=-20, n::Int64=256, noverlap::Int64=0, fs::Int64=16000)

    # load data file
    yi, fsi = wavread(file, format="double")

    # convert sampling frequency to float
    fsi = convert(Float64, fsi)

    # get mono sound
    yi = squeeze(yi[:,1])

    # resample signal and crop
    yi = resample(yi, fs/fsi)
    yi = yi[1*fs:(1+duration)*fs-1]
    yi = yi + sqrt(10^(noise_db/10)) .* randn(length(yi))
    
    # calculate frequency and log-power spectrum
    Yi = stft(yi, n, noverlap)
    logY2i = log.(abs2.(Yi))

    # return data
    return yi, Yi, logY2i
end

function squeeze(A::AbstractArray{T,N}) where {T,N}

    # find singleton dimensions
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)

    A = dropdims(A; dims=singleton_dims)

    # return array with dropped dimensions
    return A

end