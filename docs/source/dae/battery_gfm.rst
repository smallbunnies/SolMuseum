Battery GFM
===========

``battery_gfm`` reproduces the ``case_inverter.jl`` grid-forming battery boundary from PSD with
``FixedDCSource + KauraPLL + VirtualInertia + ReactivePowerDroop + VoltageModeControl + AverageConverter + LCLFilter``.
It intentionally does not add SOC or electrochemical ODEs in this module.

.. autoclass:: SolMuseum.dae.battery_gfm.battery_gfm
   :members:
