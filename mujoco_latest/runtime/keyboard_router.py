"""Normalize Kit enum and RTC text events without simulator dependencies."""
class KeyboardRouter:
    def __init__(self,controller,reopen):
        self.controller,self.reopen=controller,reopen
        self.held=set()
    def poll(self,read_down):
        """Renew native input only when the device confirms a key is still down.

        WebRTC may emit just one press and one release, with no repeat events.
        Stored events alone must not renew a lease after device loss or STOP.
        """
        source='webrtc-keyboard'
        motion=self.held & set('WASDQE')
        if self.controller.source!=source:
            self.held.difference_update('WASDQE')
            return False
        if not motion:return False
        try:down={key for key in motion if read_down(key)}
        except Exception:
            # No fresh device evidence: keep the existing controller timeout.
            self.controller.stats['keyboard_query_errors']+=1
            return False
        self.held.difference_update(motion-down)
        if self.controller.feed(source,down,confirmed_held=True):
            self.controller.stats['keyboard_state_refreshes']+=1
            return True
        return False
    def handle(self,event):
        raw=event.input
        name=str(getattr(raw,'name',raw)).upper()
        name={'UP':'W','DOWN':'S','LEFT':'A','RIGHT':'D','ARROW_UP':'W','ARROW_DOWN':'S',
              'ARROW_LEFT':'A','ARROW_RIGHT':'D'}.get(name,name)
        kind=str(getattr(event.type,'name',event.type)).upper().split('.')[-1]
        release=kind=='KEY_RELEASE'
        press=kind in ('KEY_PRESS','KEY_REPEAT')
        text=kind in ('CHAR','CHARACTER','TEXT','CHAR_INPUT')
        if name=='F1' and kind=='KEY_PRESS':self.reopen();return True
        if name not in set('WASDQEHPURMK') or not (release or press or text):return False
        source='webrtc-keyboard'
        if self.controller.source!='webrtc-keyboard':
            self.held.difference_update('WASDQE')
        if release:
            self.held.discard(name)
            self.controller.feed(source,self.held & set('WASDQE'),confirmed_held=True)
        elif name in 'HPURMK':
            if kind=='KEY_PRESS' and name not in self.held:
                self.held.add(name)
                self.controller.feed(source,(self.held & set('WASDQE'))|{name},confirmed_held=True)
                self.controller.feed(source,self.held & set('WASDQE'),confirmed_held=True)
        else:
            # Text/repeat events also renew input. poll() independently checks
            # device state so holding a key does not require OS auto-repeat.
            self.held.add(name)
            self.controller.feed(source,self.held & set('WASDQE'),confirmed_held=True)
        return True
