#ifndef GTK_EXPANDER_HPP
#define GTK_EXPANDER_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::Expander
 */
class Expander
    :
    public Gtk::CoreGObject
{
public:

    Expander( const Falcon::CoreClass*, const GtkExpander* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC new_with_mnemonic( VMARG );

    static FALCON_FUNC set_expanded( VMARG );

    static FALCON_FUNC get_expanded( VMARG );

    static FALCON_FUNC set_spacing( VMARG );

    static FALCON_FUNC get_spacing( VMARG );

    static FALCON_FUNC set_label( VMARG );

    static FALCON_FUNC get_label( VMARG );

    static FALCON_FUNC set_use_underline( VMARG );

    static FALCON_FUNC get_use_underline( VMARG );

    static FALCON_FUNC set_use_markup( VMARG );

    static FALCON_FUNC get_use_markup( VMARG );

    static FALCON_FUNC set_label_widget( VMARG );

    static FALCON_FUNC get_label_widget( VMARG );

};


} // Gtk
} // Falcon

#endif // !GTK_EXPANDER_HPP
