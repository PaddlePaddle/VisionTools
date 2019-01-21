local cv = require("luac_cv")
local basic = require("luac_basic")

function lua_main(sample, str_sample)
    local img = sample[1]
    --local label = basic.mat2str(sample[2])

    local dec = cv.imdecode(img, 1)
    local resizeimg = cv.Mat()
    cv.resize(dec, resizeimg, cv.Size(224, 224), 0, 0, cv.INTER_LINEAR)

    local result = cv.Mat()
    cv.cvtColor(resizeimg, result, cv.COLOR_BGR2RGB, 0)

    return {result, sample[2]}
end
