#ifndef GTK_HSCALE_HPP
#define GTK_HSCALE_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::HScale
 */
class HScale
    :
    public Gtk::CoreGObject
{
public:

    HScale( const Falcon::CoreClass*, const GtkHScale* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_with_range( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_HSCALE_HPP
