#ifndef GTK_HBOX_HPP
#define GTK_HBOX_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::HBox
 */
class HBox
    :
    public Gtk::CoreGObject
{
public:

    HBox( const Falcon::CoreClass*, const GtkHBox* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_HBOX_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
