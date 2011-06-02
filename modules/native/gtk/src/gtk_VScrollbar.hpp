#ifndef GTK_VSCROLLBAR_HPP
#define GTK_VSCROLLBAR_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::VScrollbar
 */
class VScrollbar
    :
    public Gtk::CoreGObject
{
public:

    VScrollbar( const Falcon::CoreClass*, const GtkVScrollbar* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_VSCROLLBAR_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
