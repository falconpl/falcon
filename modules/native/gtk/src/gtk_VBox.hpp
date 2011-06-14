#ifndef GTK_VBOX_HPP
#define GTK_VBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::VBox
 */
class VBox
    :
    public Gtk::CoreGObject
{
public:

    VBox( const Falcon::CoreClass*, const GtkVBox* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_VBOX_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
