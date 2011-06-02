#ifndef GTK_RULER_HPP
#define GTK_RULER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Ruler
 */
class Ruler
    :
    public Gtk::CoreGObject
{
public:

    Ruler( const Falcon::CoreClass*, const GtkRuler* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC set_metric( VMARG );

    static FALCON_FUNC set_range( VMARG );

    static FALCON_FUNC get_metric( VMARG );

    static FALCON_FUNC get_range( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_RULER_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
