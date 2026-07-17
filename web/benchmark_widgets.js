import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "BenchmarkNodes.CascadingSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "BenchmarkSelector") return;

        async function updateVariants(testGroupWidget, variantWidget, testGroup) {
            if (!testGroup) return;
            try {
                const resp = await fetch(
                    `/benchmark-nodes/variants?test=${encodeURIComponent(testGroup)}`
                );
                if (!resp.ok) {
                    console.warn(`[BenchmarkNodes] /benchmark-nodes/variants returned ${resp.status}`);
                    return;
                }
                const data = await resp.json();
                const variants = data.variants ?? [];
                if (variants.length === 0) return;

                const current = variantWidget.value;
                variantWidget.options.values = variants;
                variantWidget.value = variants.includes(current) ? current : variants[0];

                app.graph.setDirtyCanvas(true, true);
            } catch (err) {
                console.error("[BenchmarkNodes] Failed to fetch variants:", err);
            }
        }

        // onNodeCreated fires before saved widget values are restored — only wire
        // up the user-change callback here, do NOT call updateVariants yet.
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const testGroupWidget = this.widgets?.find(w => w.name === "test_group");
            const variantWidget   = this.widgets?.find(w => w.name === "variant");

            if (!testGroupWidget || !variantWidget) return result;

            const origCallback = testGroupWidget.callback;
            testGroupWidget.callback = async function (value) {
                origCallback?.call(this, value);
                await updateVariants(testGroupWidget, variantWidget, value);
            };

            return result;
        };

        // onConfigure fires after widget values are restored from saved workflow data,
        // so testGroupWidget.value is the correct saved test group at this point.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = async function (info) {
            const result = onConfigure?.apply(this, arguments);

            const testGroupWidget = this.widgets?.find(w => w.name === "test_group");
            const variantWidget   = this.widgets?.find(w => w.name === "variant");

            if (testGroupWidget && variantWidget) {
                await updateVariants(testGroupWidget, variantWidget, testGroupWidget.value);
            }

            return result;
        };
    },
});
