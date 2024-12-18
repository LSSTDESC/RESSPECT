# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 14 April 2020
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from resspect.plot_results import Canvas

__all__ = ['make_metrics_plots']


def make_metrics_plots(user_input):
    """Generate metric plots.

    Parameters
    ----------
    -m: list
        List of paths to the metrics files. One path for each
        learning strategy we wish to plot. This must follow the
        same order as give in `-s`.
    -o: str
        Path to output file where the plot will be stored.
    -s: list
        List of keywords describing the learning strategies to be plotted.
        Order must be the same as provided in `-m`.
        Options are ['canonical', 'rand_sampling', 'unc_sampling']

    Examples
    --------
    Use it directly from the command line.
    For example, if you wish to make a metric plot for the random sampling and
    uncertainty sampling together, do:

    >>> make_metrics_plots.py -m <path to rand sampling metrics> <path to unc sampling metrics>
    >>>     -o <path to output plot file> -s RandomSampling UncSampling

    """

    # create Canvas object
    cv = Canvas()

    # load data
    cv.load_metrics(path_to_files=list(user_input.metrics),
                        strategies_list=list(user_input.strategies))
    
    # set plot dimensions
    cv.set_plot_dimensions()

    # save plot to file
    cv.plot_metrics(output_plot_file=user_input.output,
                        strategies_list=list(user_input.strategies),
                        lim_queries=user_input.lim_queries)


def main():

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='resspect - '
                                                 'Learn loop module')
    parser.add_argument('-m', '--metrics-files-list', dest='metrics',
                        required=True, type=str,
                        help='List of path to metrics files for '
                             'different learning strategies.', nargs='+')
    parser.add_argument('-o', '--output-plot', dest='output',
                        required=True, type=str,
                        help='Complete path to output plot file.')
    parser.add_argument('-s', '--strategy-names-list', dest='strategies',
                        required=True, type=str, nargs='+',
                        help='List of strategies names. This should be'
                             'in the same order as the given list of '
                             'metrics files.')
    parser.add_argument('-q', '--max-queries', dest='lim_queries', 
                        required=False, type=int, default=1000, 
                        help='Max number of queries to be plotted.')

    from_user = parser.parse_args()

    make_metrics_plots(from_user)


if __name__ == '__main__':
    main()