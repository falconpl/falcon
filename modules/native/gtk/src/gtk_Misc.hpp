#ifndef GTK_MISC_HPP
#define GTK_MISC_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Misc
 */
class Misc
    :
    public Gtk::CoreGObject
{
public:

    Misc( const Falcon::CoreClass*, const GtkMisc* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC set_alignment( VMARG );

    static FALCON_FUNC set_padding( VMARG );

    static FALCON_FUNC get_alignment( VMARG );

    static FALCON_FUNC get_padding( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_MISC_HPP
