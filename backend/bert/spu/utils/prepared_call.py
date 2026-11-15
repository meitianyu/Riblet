class PreparedCall:
    def __init__(self, run, *, compilation_seconds, argument_seconds):
        self._run=run
        self.compilation_seconds=float(compilation_seconds)
        self.argument_seconds=float(argument_seconds)
        self.consumed=False

    def execute(self):
        if self.consumed:
            raise RuntimeError('prepared secure request was already consumed')
        self.consumed=True
        run,self._run=self._run,None
        return run()
