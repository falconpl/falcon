#ifndef GTK_BIN_HPP
#define GTK_BIN_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Bin
 */
class Bin
    :
    public Gtk::CoreGObject
{
public:

    Bin( const Falcon::CoreClass*, const GtkBin* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC get_child( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_BIN_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
