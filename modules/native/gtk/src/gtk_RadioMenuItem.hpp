#ifndef GTK_RADIOMENUITEM_HPP
#define GTK_RADIOMENUITEM_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::RadioMenuItem
 */
class RadioMenuItem
    :
    public Gtk::CoreGObject
{
public:

    RadioMenuItem( const Falcon::CoreClass*, const GtkRadioMenuItem* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC signal_group_changed( VMARG );

    static void on_group_changed( GtkRadioMenuItem*, gpointer );

    static FALCON_FUNC new_with_label( VMARG );

    static FALCON_FUNC new_with_mnemonic( VMARG );
#if 0 // TODO
    static FALCON_FUNC set_group( VMARG );

    static FALCON_FUNC get_group( VMARG );
#endif

};


} // Gtk
} // Falcon

#endif // !GTK_RADIOMENUITEM_HPP

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
