local cv = require("luac_cv")
local basic = require("luac_basic")

function lua_main(sample, str_sample)
    local img = sample[1]
    -- print('fake mat:', fake_mat:total())
    local label = basic.mat2str(sample[2])
    -- print('content of labelmat:', label)
    -- print('sizeof img:', string.len(img))
    -- local img_mat = cv.Mat(1, string.len(img), cv.CV_8U, img)

    local dec = cv.imdecode(img, 1)
    local resizeimg = cv.Mat()
    cv.resize(dec, resizeimg, cv.Size(224, 224), 0, 0, cv.INTER_LINEAR)

    local result = cv.Mat()
    cv.cvtColor(resizeimg, result, cv.COLOR_BGR2RGB, 0)
    -- print(result.cols, result.rows, result:channels(), result:total())
    --return {dec, sample[2]}
    --
    --local chw_img = basic.tochw(result)

    -- print(chw_img.cols, chw_img.rows, chw_img:channels(), chw_img:total())
    return {result, sample[2]}

end
