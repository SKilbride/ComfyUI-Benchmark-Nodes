class ConfigOverride:
    """
    Optional override for the latent dimensions used during a benchmark run.
    Leave any field at -1 to keep the testconfig.json default for that dimension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "width": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 16384,
                        "tooltip": "-1 = use testconfig default",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 16384,
                        "tooltip": "-1 = use testconfig default",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 64,
                        "tooltip": "-1 = use testconfig default",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONFIG_OVERRIDE",)
    RETURN_NAMES = ("config_override",)
    FUNCTION = "create_override"
    CATEGORY = "Benchmark"

    def create_override(self, width=-1, height=-1, batch_size=-1):
        return ({"width": width, "height": height, "batch_size": batch_size},)
