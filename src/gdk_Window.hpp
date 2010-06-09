#ifndef GDK_WINDOW_HPP
#define GDK_WINDOW_HPP

#include "modgtk.hpp"


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
