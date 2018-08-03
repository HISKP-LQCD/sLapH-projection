import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import bokeh.palettes
from bokeh.models import ColumnDataSource, Whisker, HoverTool
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column

import sys


# TODO: Symmetrization and Antisymmetrization. Take negative eigenvalues under
# time reversal into account
# TODO: this does not work if I work with the column indices directly rather
# than the config numbers. Using np.random.randint and iloc should be
# considerably faster
def bootstrap(df, bootstrapsize):
    """
  Apply the bootstrap method to randomly resample gauge configurations

  Parameters
  ----------
  df : pd.DataFrame
      Lattice data with arbitrary rows and columns cnfg x T.
  bootstrapsize : int
      The number of bootstrap samles being drawn from `df`

  Returns
  -------
  boot : pd.DataFrame
      Lattice data with rows like `df` and columns boot x T. The number of
      level0 column entries is `bootstrapsize` and it contains the mean of
      nb_cnfg randomly drawn configurations.
  """
    np.random.seed(1227)
    idx = pd.IndexSlice

    # list of all configuration numbers
    cnfgs = df.columns.levels[0].values
    rnd_samples = np.array(
        [np.random.choice(cnfgs,
                          len(cnfgs) * bootstrapsize)]).reshape((bootstrapsize,
                                                                 len(cnfgs)))
    # writing the mean value in the first sample
    rnd_samples[0] = cnfgs

    # Also I have to use a list comprehension rather than rnd directly...
    boot = pd.concat(
        [
            df.loc[:, idx[[r
                           for r in rnd]]].mean(axis=1, level=1)
            for rnd in rnd_samples
        ],
        axis=1,
        keys=range(bootstrapsize))

    return boot


def mean_and_std(df, bootstrapsize):
    """
  Mean and standard deviation over all configurations/bootstrap samples

  Parameters
  ----------
  df : pd.DataFrame
      Table with purely real entries (see Notes) and hierarchical columns where
      level 0 is the gauge configuration/bootrstrap sample number and level 1
      is the lattice time
  bootstrapsize : int
      The number of bootstrap samles being drawn from `df`

  Returns:
  --------
  pd.DataFrame
      Table with identical indices as `df` The column level 0 is replaced by
      mean and std while level 1 remains unchanged

  Notes
  -----
  Must be taken for real and imaginary part seperately because that apparently
  breaks pandas.
  """

    boot = bootstrap(df, bootstrapsize)
    mean = boot[0]
    std = boot.std(axis=1, level=1)

    return pd.concat([mean, std], axis=1, keys=['mean', 'std'])


def p_and_gammas(p, data):
    """
  Plot all rows of given pd.DataFrame into a single page as seperate graphs

  data : pd.DataFrame

      Table with any quantity as rows and multicolumns where level 0 contains
      {'mean', 'std'} and level 1 contains 'T'

  label_template : string

      Format string that will be used to label the graphs. The format must fit
      the index of `data`
  """

    row_names = data.index.names
    row_values = data.index.values
    data = data.T.unstack().T

    # iterrows() returns a tuple (index, series)
    # index is a string (the index of data must be strings for this to work). In
    # data has a MultiIndex, index is a tuple of strings
    # series contains the mean and std for every timeslice
    for counter, rv in enumerate(row_values):

        df = data.ix[rv]
        df['upper'] = df['mean'] + df['std']
        df['lower'] = df['mean'] - df['std']

        for name, value in zip(row_names, rv):
            df[name] = value
        df.rename(columns={'p^{0}_{so}': 'p0so',
                           'p^{1}_{so}': 'p1so',
                           'p^{0}_{si}': 'p0si',
                           'p^{1}_{si}': 'p1si',
                           '\gamma^{0}_{so}': 'g0so',
                           '\gamma^{1}_{so}': 'g1so',
                           '\gamma^{0}_{si}': 'g0si',
                           '\gamma^{1}_{si}': 'g1si'}, inplace=True)

#        # prepare parameters for plot design
        if len(rv) == 1:
            cmap_brg = ['red']
        else:
            cmap_brg = bokeh.palettes.viridis(len(row_values))
#        shift = 2. / 5 / len(rows)

        w = Whisker(source=ColumnDataSource(df), base="T", upper="upper", lower="lower",
                    line_color=cmap_brg[counter])
        w.upper_head.line_color = cmap_brg[counter]
        w.lower_head.line_color = cmap_brg[counter]
        p.add_layout(w)

        p.circle(x='T', y='mean', source=ColumnDataSource(df), color=cmap_brg[counter])

#        # plot
#        plt.errorbar(
#            T - 1./5 + shift * counter,
#            mean,
#            std,
#            fmt=symbol[counter % len(symbol)],
#            color=cmap_brg[counter],
#            label=label,
#            markersize=3,
#            capsize=3,
#            capthick=0.5,
#            elinewidth=0.5,
#            markeredgecolor=cmap_brg[counter],
#            linewidth='0.0')


def experimental(plotdata, diagram, bootstrapsize, name, logscale=True, verbose=False):
    """
    Create a multipage plot with a page for every element of the rho gevp. Each
    page contains one graph for each row of the irrep, summed over all momenta.

    Parameters
    ----------

    plotdata : pd.DataFrame

        Table with physical quantum numbers as rows 'gevp_row' x 'gevp_col' x
        'p_{cm}' x \mu' '\gamma_{so}' x '\gamma_{si}' and columns 'cnfg' x 'T'.
        Contains the linear combinations of correlation functions transforming
        like what the parameters qn_irrep was created with, i.e. the subduced
        data

    diagram : string
        The diagram as it will appear in the plot labels.

    bootstrapsize : int

        The number of bootstrap samples being drawn from `gevp_data`.

    pdfplot : mpl.PdfPages object

        Plots will be written to the path `pdfplot` was created with.

    See also
    --------

    utils.create_pdfplot()
    """

    # output to static HTML file
    output_file(name)

    plotdata = mean_and_std(plotdata, bootstrapsize)

    # abs of smallest positive value
    linthreshy = plotdata['mean'][plotdata['mean'] > 0].min().min()
    # abs of value closest to zero
    #linthreshy = plotdata['mean'].iloc[plotdata.loc[:,('mean',0)].nonzero()].abs().min().min()

    # list for subplots
    ss = []

    # create list of gevp elements to loop over
    plotlabel = list(set([(i[0], i[1], i[2], i[3]) for i in plotdata.index.values]))
    for graphlabel in plotlabel:

        if verbose:
            print '\tplotting p_cm = ', graphlabel[2], ', \mu = ', graphlabel[3]

        # prepare data to plot
        graphdata = plotdata.xs(
            graphlabel,
            level=[
                'gevp_row',
                'gevp_col',
                'p_{cm}',
                '\mu'])

        # prepare plot
        title = r'Gevp Element ${}$ - ${}$, $\vec{{P}}_\textnormal{{cm}} = {}$, $\mu = {}$'.format(
            graphlabel[0], graphlabel[1], graphlabel[2], graphlabel[3])

        # TODO: add color to hovertool
        hover = HoverTool(tooltips=[
            ("T", "@T"),
            ("p^{0}_{so}", "@{p0so}"),
            ("p^{1}_{so}", "@{p1so}"),
            ("p^{0}_{si}", "@{p0si}"),
            ("p^{1}_{si}", "@{p1si}"),
            ("\gamma^{0}_{so}", "@{g0so}"),
            ("\gamma^{1}_{so}", "@{g1so}"),
            ("\gamma^{0}_{si}", "@{g0si}"),
            ("\gamma^{1}_{si}", "@{g1si}")])

        # create a new plot
        ss.append(figure(
            tools=[hover],
            y_axis_type="linear", title=title,
            x_axis_label=r'$t/a$', y_axis_label=r'${}(t/a)$'.format(diagram)
        )
        )

        # plot
        p_and_gammas(ss[-1], graphdata)

    show(column(ss))
