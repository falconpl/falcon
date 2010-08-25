#ifndef GTK_HSCROLLBAR_HPP
#define GTK_HSCROLLBAR_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::HScrollbar
 */
class HScrollbar
    :
    public Gtk::CoreGObject
{
public:

    HScrollbar( const Falcon::CoreClass*, const GtkHScrollbar* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_HSCROLLBAR_HPP
