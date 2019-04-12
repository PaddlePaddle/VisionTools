import copy
import traceback
from ..source import source


class ReaderSetting(object):
    """ settings to build a reader for 'train|val|test'
    """

    def __init__(self, uri, sc_setting=None, pl_setting=None):
        """ init

        Args:
            @uri (str): data location uri, eg: file://path/to/your/data
            @sc_setting (dict): settings for load a 'source' from 'uri'
            @pl_setting (dict): settings for customizing transformation pipeline
        """
        assert type(uri) is str and len(
            uri) > 0, 'invalid uri[%s] for ReaderSetting' % (uri)
        self.uri = uri
        self.sc_setting = copy.deepcopy(sc_setting)
        self.pl_setting = copy.deepcopy(pl_setting)


class ReaderBuilder(object):
    """ Helper class to facilitate the reader building process
        which usually consists of 'train' 'val' and 'test' readers.

        Notes:
            1, This builder will first build 'source' for each reader using 'settings[xxx].sc_setting'
                if no 'sources' provided
            2, Then apply a customized transformations using 'settings[xxx].pl_setting' 
                to this source
            3, Return the transformed reader
    """

    def __init__(self, settings, pl_name='imagenet', sources=None):
        """ init

        Args:
            settings (dict): ReaderSettings 'train|val|test' reader
            pl_name (str): name of pipeline used to transform data
        """
        for k, v in settings.items():
            assert isinstance(
                v, ReaderSetting), 'invalid settings for ReaderBuilder'

        self.settings = settings
        self.pl_name = pl_name
        self.sources = sources
        if sources is None:
            self.sources = {}

    def get_source(self, which, rd_setting=None):
        """ get source for the 'which' reader 

        Args:
            @which (str): which source to get, eg: train|val|test
            @rd_setting (ReaderSetting): must provide is to to create the source

        Return:
            source of the reader
        """
        if which in self.sources:
            return self.sources[which]
        else:
            assert rd_setting is not None, 'no ReaderSetting provided for source[%s]' % (
                which)

        uri = rd_setting.uri
        setting = rd_setting.sc_setting
        if setting is None:
            setting = {}

        sc = source.load(uri=uri, **setting)
        self.sources[which] = sc
        return sc

    def _build(self, which):
        assert which in self.settings, 'not found ReaderSetting for data[%s]' % (
            which)
        rd_setting = self.settings[which]
        sc = self.get_source(which, rd_setting)
        pl_setting = rd_setting.pl_setting
        mod_name = '.'.join([self.pl_name] * 2)
        try:
            mod = __import__(mod_name, globals(), locals())
            mod = getattr(mod, self.pl_name)
            func = getattr(mod, which)
            pl = func(pl_setting)
        except Exception as e:
            stack_info = traceback.format_exc()
            raise ValueError('invalid name of pipeline[%s] with stack[%s]' %
                             (self.pl_name, stack_info))

        return pl.transform(sc.reader())

    def train(self):
        """ build reader for model training
        """
        return self._build('train')

    def val(self, settings=None):
        """ build reader for model validation
        """
        return self._build('val')

    def test(self, settings=None):
        """ build reader for model testing
        """
        return self._build('test')
