#ifndef GDK_WINDOW_HPP
#define GDK_WINDOW_HPP

#include "modgtk.hpp"

#define GET_GDKWINDOW( item ) \
        ((GdkWindow*) Falcon::dyncast<Gdk::Window*>( (item).asObjectSafe() )->getGObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Window
 */
class Window
    :
    public Gtk::CoreGObject
{
public:

    Window( const Falcon::CoreClass*, const GdkWindow* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    //static FALCON_FUNC init( VMARG );

};


} // Gdk
} // Falcon

#endif // !GDK_WINDOW_HPP
