#ifndef GTK_VSCALE_HPP
#define GTK_VSCALE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::VScale
 */
class VScale
    :
    public Gtk::CoreGObject
{
public:

    VScale( const Falcon::CoreClass*, const GtkVScale* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_with_range( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_VSCALE_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
