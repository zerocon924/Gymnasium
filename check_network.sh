#!/bin/bash
echo "=============================="
echo "  网络连通性诊断"
echo "=============================="

echo ""
echo "1. 检查常见代理端口..."
for port in 7890 7891 1080 1087 10809 10810 8080 8118; do
    if nc -z -w 1 127.0.0.1 $port 2>/dev/null; then
        echo "   ✅ 端口 $port 有服务运行"
    fi
done

echo ""
echo "2. 检查代理相关环境变量..."
echo "   HTTP_PROXY  = ${HTTP_PROXY:-（未设置）}"
echo "   HTTPS_PROXY = ${HTTPS_PROXY:-（未设置）}"
echo "   http_proxy  = ${http_proxy:-（未设置）}"
echo "   https_proxy = ${https_proxy:-（未设置）}"
echo "   ALL_PROXY   = ${ALL_PROXY:-（未设置）}"
echo "   all_proxy   = ${all_proxy:-（未设置）}"
echo "   NO_PROXY    = ${NO_PROXY:-（未设置）}"

echo ""
echo "3. 测试直连（不走代理）..."
curl --noproxy '*' -s -o /dev/null -w "   直连 yinli.one: HTTP %{http_code} (%{time_total}s)\n" --max-time 10 https://yinli.one/v1/models 2>&1 || echo "   ❌ 直连失败"

echo ""
echo "4. 测试走系统代理..."
curl -s -o /dev/null -w "   系统代理 yinli.one: HTTP %{http_code} (%{time_total}s)\n" --max-time 10 https://yinli.one/v1/models 2>&1 || echo "   ❌ 系统代理失败"

echo ""
echo "5. 测试走 7890 代理..."
curl -x http://127.0.0.1:7890 -s -o /dev/null -w "   7890代理 yinli.one: HTTP %{http_code} (%{time_total}s)\n" --max-time 10 https://yinli.one/v1/models 2>&1 || echo "   ❌ 7890 代理失败"

echo ""
echo "=============================="
echo "  诊断完成"
echo "=============================="
