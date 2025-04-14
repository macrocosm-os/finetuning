module.exports = {
    apps: [{
        name: "finetuning-validator",
        script: "scripts/run_validator.sh", // Use the wrapper script
        autorestart: true,
        watch: false
    }]
};
