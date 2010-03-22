/**
 *  \file gtk_Widget.cpp
 */

#include "gtk_Widget.hpp"

#include "gdk_EventButton.hpp"

#include "gtk_Requisition.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {


void Widget::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Widget = mod->addClass( "GtkWidget", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkObject" ) );
    c_Widget->getClassDef()->addInheritance( in );

    c_Widget->setWKS( true );
    c_Widget->getClassDef()->factory( &Widget::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_accel_closures_changed",  &Widget::signal_accel_closures_changed },
    { "button_press_event",             &Widget::signal_button_press_event },
    //{ "button_release_event",           &Widget::signal_button_release_event },
    { "signal_can_activate_accel",      &Widget::signal_can_activate_accel },
    //{ "signal_child_notify",            &Widget::signal_child_notify },
    //{ "signal_client_event",            &Widget::signal_client_event },
    { "signal_composited_changed",      &Widget::signal_composited_changed },
    //{ "signal_configure_event",         &Widget::signal_configure_event },
    //{ "signal_damage_event",            &Widget::signal_damage_event },
    { "signal_delete_event",            &Widget::signal_delete_event },
    { "signal_destroy_event",           &Widget::signal_destroy_event },
    //{ "signal_direction_changed",       &Widget::signal_direction_changed },
    //{ "signal_drag_begin",              &Widget::signal_drag_begin },
    //{ "signal_drag_data_delete",        &Widget::signal_drag_data_delete },
    //{ "signal_drag_data_get",           &Widget::signal_drag_data_get },
    //{ "signal_drag_data_received",      &Widget::signal_drag_data_received },
    //{ "signal_drag_drop",               &Widget::signal_drag_drop },
    //{ "signal_drag_end",                &Widget::signal_drag_end },
    //{ "signal_drag_failed",             &Widget::signal_drag_failed },
    //{ "signal_drag_leave",              &Widget::signal_drag_leave },
    //{ "signal_drag_motion",             &Widget::signal_drag_motion },
    //{ "signal_enter_notify_event",      &Widget::signal_enter_notify_event },
    //{ "signal_event",                   &Widget::signal_event },
    //{ "signal_event_after",             &Widget::signal_event_after },
    //{ "signal_expose_event",            &widget::signal_expose_event },
    //{ "signal_focus",                   &Widget::signal_focus },
    //{ "signal_focus_in_event",          &Widget::signal_focus_in_event },
    //{ "signal_focus_out_event",         &Widget::signal_focus_out_event },
    //{ "signal_grab_broken_event",       &Widget::signal_grab_broken_event },
    //{ "signal_grab_focus",              &Widget::signal_grab_focus },
    //{ "signal_grab_notify",             &Widget::signal_grab_notify },
    { "signal_hide",                    &Widget::signal_hide },
    //{ "signal_hierarchy_changed",       &Widget::signal_hierarchy_changed },
    //{ "signal_key_press_event",         &Widget::signal_key_press_event },
    //{ "signal_key_release_event",       &Widget::signal_key_release_event },
    //{ "signal_keynav_failed",           &Widget::signal_keynav_failed },
    //{ "signal_leave_notify_event",      &Widget::signal_leave_notify_event },
    //{ "signal_map",                     &Widget::signal_map },
    //{ "signal_map_event",               &Widget::signal_map_event },
    //{ "signal_mnemonic_activate",       &Widget::signal_mnemonic_activate },
    //{ "signal_motion_notify_event",     &Widget::signal_motion_notify_event },
    //{ "signal_move_focus",              &Widget::signal_move_focus },
    //{ "signal_no_expose_event",         &Widget::signal_no_expose_event },
    //{ "signal_parent_set",              &Widget::signal_parent_set },
    //{ "signal_popup_menu",              &Widget::signal_popup_menu },
    //{ "signal_property_notify_event",   &Widget::signal_property_notify_event },
    //{ "signal_proximity_in_event",      &Widget::signal_proximity_in_event },
    //{ "signal_proximity_out_event",     &Widget::signal_proximity_out_event },
    //{ "signal_query_tooltip",           &Widget::signal_query_tooltip },
    //{ "signal_realize",                 &Widget::signal_realize },
    //{ "signal_screen_changed",          &Widget::signal_screen_changed },
    //{ "signal_scroll_event",            &Widget::signal_scroll_event },
    //{ "signal_selection_clear_event",   &Widget::signal_selection_clear_event },
    //{ "signal_selection_get",           &Widget::signal_selection_get },
    //{ "signal_selection_notify_event",  &Widget::signal_selection_notify_event },
    //{ "signal_selection_received",      &Widget::signal_selection_received },
    //{ "signal_selection_request_event", &Widget::signal_selection_request_event },
    { "signal_show",                    &Widget::signal_show },
    //{ "signal_show_help",               &Widget::signal_show_help },
    //{ "signal_size_allocate",           &Widget::signal_size_allocate },
    { "signal_size_request",            &Widget::signal_size_request },
    //{ "signal_state_changed",           &Widget::signal_state_changed },
    //{ "signal_style_set",               &Widget::signal_style_set },
    //{ "signal_unmap",                   &Widget::signal_unmap },
    //{ "signal_unmap_event",             &Widget::signal_unmap_event },
    //{ "signal_unrealize",               &Widget::signal_unrealize },
    //{ "signal_visibility_notify_event", &Widget::signal_visibility_notify_event },
    //{ "signal_window_state_event",      &Widget::signal_window_state_event },
    { "destroy",                &Widget::destroy },
    { "unparent",               &Widget::unparent },
    { "show",                   &Widget::show },
    { "show_now",               &Widget::show_now },
    { "hide",                   &Widget::hide },
    { "show_all",               &Widget::show_all },
    { "hide_all",               &Widget::hide_all },
    { "map",                    &Widget::map },
    { "unmap",                  &Widget::unmap },
    { "realize",                &Widget::realize },
    { "unrealize",              &Widget::unrealize },
    { "queue_draw",             &Widget::queue_draw },
    { "queue_resize",           &Widget::queue_resize },
    { "queue_resize_no_redraw", &Widget::queue_resize_no_redraw },
    //{ "draw",                   &Widget::draw },
    { "size_request",           &Widget::size_request },
    { "get_child_requisition",  &Widget::get_child_requisition },


    { "reparent",               &Widget::reparent },
    { "is_focus",               &Widget::is_focus },
    { "grab_focus",             &Widget::grab_focus },
    { "grab_default",           &Widget::grab_default },
    { "set_name",               &Widget::set_name },
    { "get_name",               &Widget::get_name },
    { "set_sensitive",          &Widget::set_sensitive },
    { "set_events",             &Widget::set_events },
    { "add_events",             &Widget::add_events },
    { "get_toplevel",           &Widget::set_sensitive },
    { "get_events",             &Widget::get_events },
    { "is_ancestor",            &Widget::is_ancestor },
    { "hide_on_delete",         &Widget::hide_on_delete },

    { "get_size_request",       &Widget::get_size_request },
    //{ "set_child_visible",      &Widget::set_child_visible },
    //{ "set_default_visual",     &Widget::set_default_visual },
    { "set_size_request",       &Widget::set_size_request },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Widget, meth->name, meth->cb );
}


Widget::Widget( const Falcon::CoreClass* gen, const GtkWidget* wdt )
    :
    Gtk::CoreGObject( gen )
{
    if ( wdt )
        setUserData( new GData( (GObject*) wdt ) );
}


Falcon::CoreObject* Widget::factory( const Falcon::CoreClass* gen, void* wdt, bool )
{
    return new Widget( gen, (GtkWidget*) wdt );
}


/*#
    @class GtkWidget
    @brief Base class for all widgets

    GtkWidget is the base class all widgets in GTK+ derive from.
    It manages the widget lifecycle, states and style.
 */


/*#
    @method signal_accel_closures_changed GtkWidget
    @brief Connect a VMSlot to the widget accel_closures_changed signal and return it
 */
FALCON_FUNC Widget::signal_accel_closures_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "accel_closures_changed",
        (void*) &Widget::on_accel_closures_changed, vm );
}


void Widget::on_accel_closures_changed( GtkWidget* wdt, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) wdt, "accel_closures_changed",
        "on_accel_closures_changed", (VMachine*)_vm );
}


/*#
    @method signal_button_press_event GtkWidget
    @brief Connect a VMSlot to the widget button_press_event signal and return it

    The button-press-event signal will be emitted when a button (typically from a mouse)
    is pressed.

    To receive this signal, the GdkWindow associated to the widget needs to enable
    the GDK_BUTTON_PRESS_MASK mask.

    This signal will be sent to the grab widget if there is one.

    Your callback function gets a GdkEventButton as parameter, and must return
    a boolean, true to stop other handlers from being invoked for the event,
    false to propagate the event further.
 */
FALCON_FUNC Widget::signal_button_press_event( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "button_press_event",
        (void*) &Widget::on_button_press_event, vm );
}


gboolean Widget::on_button_press_event( GtkWidget* obj, GdkEventButton* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "button_press_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GdkEventButton" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_button_press_event", it ) )
            {
                printf(
                "[GtkWidget::on_button_press_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        vm->pushParam( new Gdk::EventButton( wki->asClass(), ev ) );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_can_activate_accel] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


//FALCON_FUNC signal_button_release_event( VMARG );

//gboolean on_button_release_event( GtkWidget*, GdkEventButton*, gpointer );



/*#
    @method signal_can_activate_accel GtkWidget
    @brief Connect a VMSlot to the widget can_activate_accel signal and return it

    Determines whether an accelerator that activates the signal identified by
    signal_id can currently be activated. This signal is present to allow applications
    and derived widgets to override the default GtkWidget handling for determining
    whether an accelerator can be activated.

    The callback function must return a boolean (returning true if the signal can
    be activated). It will get an integer as parameter, that is the ID of a signal
    installed on widget.
 */
FALCON_FUNC Widget::signal_can_activate_accel( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "can_activate_accel",
        (void*) &Widget::on_can_activate_accel, vm );
}


gboolean Widget::on_can_activate_accel( GtkWidget* obj, guint signal_id, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "can_activate_accel", false );

    if ( !cs || cs->empty() )
        return TRUE; // can activate

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_can_activate_accel", it ) )
            {
                printf(
                "[GtkWidget::on_can_activate_accel] invalid callback (expected callable)\n" );
                return FALSE; // block event
            }
        }
        vm->pushParam( (int64) signal_id );
        vm->callItem( it, 1 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( !it.asBoolean() )
                return FALSE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_can_activate_accel] invalid callback (expected boolean)\n" );
            return FALSE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return TRUE; // can activate
}


//FALCON_FUNC Widget::signal_child_notify( VMARG );

//gboolean Widget::on_child_notify( GtkWidget*, GParamSpec*, gpointer );

//FALCON_FUNC Widget::signal_client_event( VMARG );

//gboolean Widget::on_client_event( GtkWidget*, GdkEventClient*, gpointer );


/*#
    @method signal_composited_changed GtkWidget
    @brief Connect a VMSlot to the widget composited_changed signal and return it

    The composited-changed signal is emitted when the composited status of
    widgets screen changes.
 */
FALCON_FUNC Widget::signal_composited_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "composited_changed",
        (void*) &Widget::on_composited_changed, vm );
}


void Widget::on_composited_changed( GtkWidget* obj, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) obj, "composited_changed",
        "on_composited_changed", (VMachine*)_vm );
}


//FALCON_FUNC Widget::signal_configure_event( VMARG );

//gboolean Widget::on_configure_event( GtkWidget*, GdkEventConfigure*, gpointer );

//FALCON_FUNC signal_damage_event( VMARG );

//gboolean on_damage_event( GtkWidget*, GdkEvent*, gpointer );


/*#
    @method signal_delete_event GtkWidget
    @brief Connect a VMSlot to the widget delete_event signal and return it

    The callback function must return a boolean, that if is true, will block the event.

    The delete-event signal is emitted if a user requests that a toplevel window
    is closed. The default handler for this signal destroys the window.
    Connecting gtk_widget_hide_on_delete() to this signal will cause the window
    to be hidden instead, so that it can later be shown again without reconstructing it.
 */
FALCON_FUNC Widget::signal_delete_event( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "delete_event", (void*) &Widget::on_delete_event, vm );
}


gboolean Widget::on_delete_event( GtkWidget* obj, GdkEvent* ev, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "delete_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_delete_event", it ) )
            {
                printf(
                "[GtkWidget::on_delete_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        //vm->pushParam( Item( (int64)((GdkEventAny*)ev)->type ) );
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_delete_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


/*#
    @method signal_destroy_event GtkWidget
    @brief Connect a VMSlot to the widget destroy signal and return it

    The callback function must return a boolean, true to stop other handlers from
    being invoked for the event, false to propagate the event further.

    The destroy-event signal is emitted when a GdkWindow is destroyed.
    You rarely get this signal, because most widgets disconnect themselves from
    their window before they destroy it, so no widget owns the window at destroy time.

    To receive this signal, the GdkWindow associated to the widget needs to enable
    the GDK_STRUCTURE_MASK mask. GDK will enable this mask automatically for all new windows.
 */
FALCON_FUNC Widget::signal_destroy_event( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "destroy_event", (void*) &Widget::on_destroy_event, vm );
}


gboolean Widget::on_destroy_event( GtkWidget* obj, GdkEvent*, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "destroy_event", false );

    if ( !cs || cs->empty() )
        return FALSE; // propagate event

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_destroy_event", it ) )
            {
                printf(
                "[GtkWidget::on_destroy_event] invalid callback (expected callable)\n" );
                return TRUE; // block event
            }
        }
        //vm->pushParam( Item( (int64)((GdkEventAny*)ev)->type ) );
        vm->callItem( it, 0 );
        it = vm->regA();

        if ( !it.isNil() && it.isBoolean() )
        {
            if ( it.asBoolean() )
                return TRUE; // block event
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkWidget::on_destroy_event] invalid callback (expected boolean)\n" );
            return TRUE; // block event
        }
    }
    while ( iter.hasCurrent() );

    return FALSE; // propagate event
}


//FALCON_FUNC Widget::signal_direction_changed( VMARG );

//void Widget::on_direction_changed( GtkWidget*, GtkTextDirection, gpointer );

//FALCON_FUNC Widget::signal_drag_begin( VMARG );

//void Widget::on_drag_begin( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_data_delete( VMARG );

//void Widget::on_drag_data_delete( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_data_get( VMARG );

//void Widget::on_drag_data_get( GtkWidget*, GdkDragContext*, GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_data_received( VMARG );

//void Widget::on_drag_data_received( GtkWidget*, GdkDragContext*, gint, gint,
        //GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_drop( VMARG );

//gboolean Widget::on_drag_drop( GtkWidget*, GdkDragContext*, gint, gint, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_end( VMARG );

//void Widget::on_drag_end( GtkWidget*, GdkDragContext*, gpointer );

//FALCON_FUNC Widget::signal_drag_failed( VMARG );

//gboolean Widget::on_drag_failed( GtkWidget*, GdkDragContext*, GtkDragResult, gpointer );

//FALCON_FUNC Widget::signal_drag_leave( VMARG );

//void Widget::on_drag_leave( GtkWidget*, GdkDragContext*, guint, gpointer );

//FALCON_FUNC Widget::signal_drag_motion( VMARG );

//gboolean Widget::on_drag_motion( GtkWidget*, GdkDragContext*, gint, gint, guint, gpointer );

//FALCON_FUNC Widget::signal_enter_notify_event( VMARG );

//gboolean Widget::on_enter_notify_event( GtkWidget*, GdkEventCrossing*, gpointer );

//FALCON_FUNC Widget::signal_event( VMARG );

//gboolean Widget::on_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_event_after( VMARG );

//void Widget::on_event_after( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_expose_event( VMARG );

//gboolean Widget::on_expose_event( GtkWidget*, GdkEventExpose*, gpointer );

//FALCON_FUNC Widget::signal_focus( VMARG );

//gboolean Widget::on_focus( GtkWidget*, GtkDirectionType*, gpointer );

//FALCON_FUNC Widget::signal_focus_in_event( VMARG );

//gboolean Widget::on_focus_in_event( GtkWidget*, GdkEventFocus*, gpointer );

//FALCON_FUNC Widget::signal_focus_out_event( VMARG );

//gboolean Widget::on_focus_out_event( GtkWidget*, GdkEventFocus*, gpointer );

//FALCON_FUNC Widget::signal_grab_broken_event( VMARG );

//gboolean Widget::on_grab_broken_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_grab_focus( VMARG );

//void Widget::on_grab_focus( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_grab_notify( VMARG );

//void Widget::on_grab_notify( GtkWidget*, gboolean, gpointer );


/*#
    @method signal_hide GtkWidget
    @brief Connect a VMSlot to the widget hide signal and return it
 */
FALCON_FUNC Widget::signal_hide( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "hide", (void*) &Widget::on_hide, vm );
}


void Widget::on_hide( GtkWidget* obj, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) obj, "hide", "on_hide", (VMachine*)_vm );
}




//FALCON_FUNC Widget::signal_hierarchy_changed( VMARG );

//void Widget::on_hierarchy_changed( GtkWidget*, GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_key_press_event( VMARG );

//gboolean Widget::on_key_press_event( GtkWidget*, GdkEventKey*, gpointer );

//FALCON_FUNC Widget::signal_key_release_event( VMARG );

//gboolean Widget::on_key_release_event( GtkWidget*, GdkEventKey*, gpointer );

//FALCON_FUNC Widget::signal_keynav_failed( VMARG );

//gboolean Widget::on_keynav_failed( GtkWidget*, GtkDirectionType, gpointer );

//FALCON_FUNC Widget::signal_leave_notify_event( VMARG );

//gboolean Widget::on_leave_notify_event( GtkWidget*, GdkEventCrossing*, gpointer );

//FALCON_FUNC Widget::signal_map( VMARG );

//void Widget::on_map( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_map_event( VMARG );

//gboolean Widget::on_map_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_mnemonic_activate( VMARG );

//gboolean Widget::on_mnemonic_activate( GtkWidget*, gboolean, gpointer );

//FALCON_FUNC Widget::signal_motion_notify_event( VMARG );

//gboolean Widget::on_motion_notify_event( GtkWidget*, GdkEventMotion*, gpointer );

//FALCON_FUNC Widget::signal_move_focus( VMARG );

//void Widget::on_move_focus( GtkWidget*, GtkDirectionType, gpointer );

//FALCON_FUNC Widget::signal_no_expose_event( VMARG );

//gboolean Widget::on_no_expose_event( GtkWidget*, GdkEventNoExpose*, gpointer );

//FALCON_FUNC Widget::signal_parent_set( VMARG );

//void Widget::on_parent_set( GtkWidget*, GtkObject*, gpointer );

//FALCON_FUNC Widget::signal_popup_menu( VMARG );

//gboolean Widget::on_popup_menu( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_property_notify_event( VMARG );

//gboolean Widget::on_property_notify_event( GtkWidget*, GdkEventProperty*, gpointer );

//FALCON_FUNC Widget::signal_proximity_in_event( VMARG );

//gboolean Widget::on_proximity_in_event( GtkWidget*, GdkEventProximity*, gpointer );

//FALCON_FUNC Widget::signal_proximity_out_event( VMARG );

//gboolean Widget::on_proximity_out_event( GtkWidget*, GdkEventProximity*, gpointer );

//FALCON_FUNC Widget::signal_query_tooltip( VMARG );

//gboolean Widget::on_query_tooltip( GtkWidget*, gint, gint, gboolean, GtkTooltip*, gpointer );

//FALCON_FUNC Widget::signal_realize( VMARG );

//void Widget::on_realize( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_screen_changed( VMARG );

//void Widget::on_screen_changed( GtkWidget*, GdkScreen*, gpointer );

//FALCON_FUNC Widget::signal_scroll_event( VMARG );

//gboolean Widget::on_scroll_event( GtkWidget*, GdkEventScroll*, gpointer );

//FALCON_FUNC Widget::signal_selection_clear_event( VMARG );

//gboolean Widget::on_selection_clear_event( GtkWidget*, GdkEventSelection*, gpointer );

//FALCON_FUNC Widget::signal_selection_get( VMARG );

//void Widget::on_selection_get( GtkWidget*, GtkSelectionData*, guint, guint, gpointer );

//FALCON_FUNC Widget::signal_selection_notify_event( VMARG );

//gboolean Widget::on_selection_notify_event( GtkWidget*, GdkEventSelection*, gpointer );

//FALCON_FUNC Widget::signal_selection_received( VMARG );

//void Widget::on_selection_received( GtkWidget*, GtkSelectionData*, guint, gpointer );

//FALCON_FUNC Widget::signal_selection_request_event( VMARG );

//gboolean Widget::on_selection_request_event( GtkWidget*, GdkEventSelection*, gpointer );


/*#
    @method signal_show GtkWidget
    @brief Connect a VMSlot to the widget show signal and return it
 */
FALCON_FUNC Widget::signal_show( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    Gtk::internal_get_slot( "show", (void*) &Widget::on_show, vm );
}


void Widget::on_show( GtkWidget* obj, gpointer _vm )
{
    Gtk::internal_trigger_slot( (GObject*) obj, "show", "on_show", (VMachine*)_vm );
}


//FALCON_FUNC Widget::signal_show_help( VMARG );

//gboolean Widget::on_show_help( GtkWidget*, GtkWidgetHelpType, gpointer );

//FALCON_FUNC Widget::signal_size_allocate( VMARG );

//void Widget::on_size_allocate( GtkWidget*, GtkAllocation*, gpointer );


/*#
    @method signal_size_request GtkWidget
    @brief Connect a VMSlot to the widget size-request signal and return it
 */
FALCON_FUNC Widget::signal_size_request( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "size_request", (void*) &Widget::on_size_request, vm );
}


void Widget::on_size_request( GtkWidget* obj, GtkRequisition* req, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "size_request", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*)_vm;
    Item* wki = vm->findWKI( "GtkRequisition" );
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();
        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_size_request", it ) )
            {
                printf(
                "[GtkWidget::on_size_request] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::Requisition( wki->asClass(), req ) );
        vm->callItem( it, 1 );
        iter.next();
    }
    while ( iter.hasCurrent() );
}


//FALCON_FUNC Widget::signal_state_changed( VMARG );

//void Widget::on_state_changed( GtkWidget*, GtkState, gpointer );

//FALCON_FUNC Widget::signal_style_set( VMARG );

//void Widget::on_style_set( GtkWidget*, GtkStyle*, gpointer );

//FALCON_FUNC Widget::signal_unmap( VMARG );

//void Widget::on_unmap( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_unmap_event( VMARG );

//gboolean Widget::on_unmap_event( GtkWidget*, GdkEvent*, gpointer );

//FALCON_FUNC Widget::signal_unrealize( VMARG );

//void Widget::on_unrealize( GtkWidget*, gpointer );

//FALCON_FUNC Widget::signal_visibility_notify_event( VMARG );

//gboolean Widget::on_visibility_notify_event( GtkWidget*, GdkEventVisibility*, gpointer );

//FALCON_FUNC Widget::signal_window_state_event( VMARG );

//gboolean Widget::on_window_state_event( GtkWidget*, GdkEventWindowState*, gpointer );



/*#
    @method destroy GtkWidget
    @brief Destroys a widget.

    Equivalent to gtk_object_destroy().

    In most cases, only toplevel widgets (windows) require explicit destruction,
    because when you destroy a toplevel its children will be destroyed as well.

    In Falcon, you should not need to use that method.
 */
FALCON_FUNC Widget::destroy( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_destroy( (GtkWidget*)_obj );
}


/*#
    @method unparent GtkWidget
    @brief This function is only for use in widget implementations.

    Should be called by implementations of the remove method on GtkContainer,
    to dissociate a child from the container.
 */
FALCON_FUNC Widget::unparent( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_unparent( (GtkWidget*)_obj );
}


/*#
    @method show GtkWidget
    @brief Flags a widget to be displayed.
 */
FALCON_FUNC Widget::show( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show( ((GtkWidget*)_obj) );
}


/*#
    @method show_now GtkWidget
    @brief Shows a widget.
 */
FALCON_FUNC Widget::show_now( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show_now( ((GtkWidget*)_obj) );
}


/*#
    @method hide GtkWidget
    @brief Reverses the effects of show(), causing the widget to be hidden (invisible to the user).
 */
FALCON_FUNC Widget::hide( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_hide( ((GtkWidget*)_obj) );
}


/*#
    @method show_all GtkWidget
    @brief Recursively shows a widget, and any child widgets (if the widget is a container).
 */
FALCON_FUNC Widget::show_all( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_show_all( ((GtkWidget*)_obj) );
}


/*#
    @method hide_all GtkWidget
    @brief Recursively hides a widget and any child widgets.
 */
FALCON_FUNC Widget::hide_all( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_hide_all( ((GtkWidget*)_obj) );
}


/*#
    @method map GtkWidget
    @brief This function is only for use in widget implementations.

    Causes a widget to be mapped if it isn't already.
 */
FALCON_FUNC Widget::map( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_map( (GtkWidget*)_obj );
}


/*#
    @method unmap GtkWidget
    @brief This function is only for use in widget implementations.

    Causes a widget to be unmapped if it's currently mapped.
 */
FALCON_FUNC Widget::unmap( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_unmap( (GtkWidget*)_obj );
}


/*#
    @method realize GtkWidget
    @brief Creates the GDK (windowing system) resources associated with a widget.

    For example, widget->window will be created when a widget is realized.
    Normally realization happens implicitly; if you show a widget and all its parent
    containers, then the widget will be realized and mapped automatically.

    Realizing a widget requires all the widget's parent widgets to be realized;
    calling gtk_widget_realize() realizes the widget's parents in addition to widget
    itself. If a widget is not yet inside a toplevel window when you realize it,
    bad things will happen.

    This function is primarily used in widget implementations, and isn't very
    useful otherwise. Many times when you think you might need it, a better
    approach is to connect to a signal that will be called after the widget is
    realized automatically, such as GtkWidget::expose-event.
    Or simply g_signal_connect() to the GtkWidget::realize signal.
 */
FALCON_FUNC Widget::realize( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_realize( (GtkWidget*)_obj );
}


/*#
    @method unrealize GtkWidget
    @brief This function is only useful in widget implementations.

    Causes a widget to be unrealized (frees all GDK resources associated with
    the widget, such as widget->window).
 */
FALCON_FUNC Widget::unrealize( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_unrealize( (GtkWidget*)_obj );
}


/*#
    @method queue_draw GtkWidget
    @brief Equivalent to calling widget.queue_draw_area() for the entire area of a widget.
 */
FALCON_FUNC Widget::queue_draw( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_queue_draw( (GtkWidget*)_obj );
}


/*#
    @method queue_resize GtkWidget
    @brief This function is only for use in widget implementations.

    Flags a widget to have its size renegotiated; should be called when a widget
    for some reason has a new size request. For example, when you change the text
    in a GtkLabel, GtkLabel queues a resize to ensure there's enough space for
    the new text.
 */
FALCON_FUNC Widget::queue_resize( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_queue_resize( (GtkWidget*)_obj );
}


/*#
    @method queue_resize_no_redraw GtkWidget
    @brief This function works like gtk_widget_queue_resize(), except that the widget is not invalidated.
*/
FALCON_FUNC Widget::queue_resize_no_redraw( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_queue_resize_no_redraw( (GtkWidget*)_obj );
}


//FALCON_FUNC Widget::draw( VMARG );


/*#
    @method size_request GtkWidget
    @brief Get the size "requisition" of the widget.
    @return GtkRequisition object

    This function is typically used when implementing a GtkContainer subclass.
    Obtains the preferred size of a widget. The container uses this information
    to arrange its child widgets and decide what size allocations to give them
    with gtk_widget_size_allocate().

    You can also call this function from an application, with some caveats.
    Most notably, getting a size request requires the widget to be associated
    with a screen, because font information may be needed. Multihead-aware
    applications should keep this in mind.

    Also remember that the size request is not necessarily the size a widget
    will actually be allocated.
 */
FALCON_FUNC Widget::size_request( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkRequisition req;
    gtk_widget_size_request( (GtkWidget*)_obj, &req );
    Item* wki = vm->findWKI( "GtkRequisition" );
    vm->retval( new Gtk::Requisition( wki->asClass(), &req ) );
}


/*#
    @method get_child_requisition GtkWidget
    @brief This function is only for use in widget implementations.

    Obtains widget->requisition, unless someone has forced a particular geometry
    on the widget (e.g. with gtk_widget_set_size_request()), in which case it
    returns that geometry instead of the widget's requisition.
 */
FALCON_FUNC Widget::get_child_requisition( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkRequisition req;
    gtk_widget_get_child_requisition( (GtkWidget*)_obj, &req );
    Item* wki = vm->findWKI( "GtkRequisition" );
    vm->retval( new Gtk::Requisition( wki->asClass(), &req ) );
}


//FALCON_FUNC Widget::size_allocate( VMARG );

//FALCON_FUNC Widget::add_accelerator( VMARG );

//FALCON_FUNC Widget::remomve_accelerator( VMARG );

//FALCON_FUNC Widget::set_accel_path( VMARG );

//FALCON_FUNC Widget::list_accel_closures( VMARG );

//FALCON_FUNC Widget::can_activate_accel( VMARG );

//FALCON_FUNC Widget::event( VMARG );



/*#
    @method activate GtkWidget
    @brief For widgets that can be "activated" (buttons, menu items, etc).
    @return boolean
 */
FALCON_FUNC Widget::activate( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_widget_activate( ((GtkWidget*)_obj) ) );
}


/*#
    @method reparent GtkWidget
    @brief Moves a widget from one container to another.
    @param new_parent The new parent
 */
FALCON_FUNC Widget::reparent( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !IS_DERIVED( i_wdt, GtkWidget ) )
    {
        throw_inv_params( "GtkWidget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    gtk_widget_reparent( (GtkWidget*)_obj, (GtkWidget*) wdt );
}


//FALCON_FUNC Widget::intersect( VMARG );



/*#
    @method is_focus GtkWidget
    @brief Determines if the widget is the focus widget within its toplevel.
    This does not mean that the HAS_FOCUS flag is necessarily set;
    HAS_FOCUS will only be set if the toplevel widget additionally has the
    global input focus.
    @return (boolean)
 */
FALCON_FUNC Widget::is_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_widget_is_focus( (GtkWidget*)_obj ) );
}


/*#
    @method grab_focus GtkWidget
    @brief Causes widget to have the keyboard focus for the GtkWindow it's inside.
    widget must be a focusable widget, such as a GtkEntry; something like GtkFrame won't work.
    More precisely, it must have the GTK_CAN_FOCUS flag set.
    Use gtk_widget_set_can_focus() to modify that flag.
 */
FALCON_FUNC Widget::grab_focus( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_grab_focus( (GtkWidget*)_obj );
}


/*#
    @method grab_default GtkWidget
    @brief Causes widget to become the default widget.
    widget must have the GTK_CAN_DEFAULT flag set; typically you have to set this flag
    yourself by calling gtk_widget_set_can_default (widget, TRUE). The default widget
    is activated when the user presses Enter in a window. Default widgets must be
    activatable, that is, gtk_widget_activate() should affect them.
 */
FALCON_FUNC Widget::grab_default( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_grab_default( (GtkWidget*)_obj );
}


/*#
    @method set_name GtkWidget
    @brief Attribute a name to the widget.
    @param name (string)
    Widgets can be named, which allows you to refer to them from a gtkrc file.
    You can apply a style to widgets with a particular name in the gtkrc file.
    See the documentation for gtkrc files (on the same page as the docs for GtkRcStyle).
    Note that widget names are separated by periods in paths (see gtk_widget_path()),
    so names with embedded periods may cause confusion.
 */
FALCON_FUNC Widget::set_name( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || i_name->isNil() || !i_name->isString() )
    {
        throw_inv_params( "S" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString s( i_name->asString() );
    gtk_widget_set_name( (GtkWidget*)_obj, s.c_str() );
}


/*#
    @method get_name GtkWidget
    @brief Get the name of the widget.
    @return (string)
    See GtkWidget.set_name() for the significance of widget names.
 */
FALCON_FUNC Widget::get_name( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* s = gtk_widget_get_name( (GtkWidget*)_obj );
    vm->retval( String( s ) );
}


//FALCON_FUNC Widget::set_state( VMARG );



/*#
    @method set_sensitive GtkWidget
    @brief Sets the sensitivity of a widget.
    A widget is sensitive if the user can interact with it.
    Insensitive widgets are "grayed out" and the user can't interact with them.
    Insensitive widgets are known as "inactive", "disabled", or "ghosted" in some other toolkits.
 */
FALCON_FUNC Widget::set_sensitive( VMARG )
{
    Item* i_sens = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_sens || i_sens->isNil() || !i_sens->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_set_sensitive( (GtkWidget*)_obj, i_sens->asBoolean() ? TRUE : FALSE );
}


//FALCON_FUNC Widget::set_parent( VMARG );

//FALCON_FUNC Widget::set_parent_window( VMARG );

//FALCON_FUNC Widget::get_parent_window( VMARG );

//FALCON_FUNC Widget::set_uposition( VMARG );

//FALCON_FUNC Widget::set_usize( VMARG );


/*#
    @method set_events GtkWidget
    @brief Sets the event mask (see GdkEventMask) for a widget.
    @param an event mask, see GdkEventMask

    The event mask determines which events a widget will receive. Keep in mind that
    different widgets have different default event masks, and by changing the event
    mask you may disrupt a widget's functionality, so be careful. This function must
    be called while a widget is unrealized. Consider add_events() for
    widgets that are already realized, or if you want to preserve the existing event
    mask. This function can't be used with GTK_NO_WINDOW widgets; to get events on
    those widgets, place them inside a GtkEventBox and receive events on the event box.
 */
FALCON_FUNC Widget::set_events( VMARG )
{
    Item* i_ev = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ev || i_ev->isNil() || !i_ev->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_set_events( (GtkWidget*)_obj, i_ev->asInteger() );
}


/*#
    @method add_events GtkWidget
    @brief Adds the events in the bitfield events to the event mask for widget.
    @param events an event mask, see GdkEventMask
 */
FALCON_FUNC Widget::add_events( VMARG )
{
    Item* i_ev = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ev || i_ev->isNil() || !i_ev->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_add_events( (GtkWidget*)_obj, i_ev->asInteger() );
}


//FALCON_FUNC Widget::set_extension_events( VMARG );

//FALCON_FUNC Widget::get_extension_events( VMARG );


/*#
    @method get_toplevel GtkWidget
    @brief This function returns the topmost widget in the container hierarchy widget is a part of.
    @return (widget)
    If widget has no parent widgets, it will be returned as the topmost widget.
    No reference will be added to the returned widget; it should not be unreferenced.
 */
FALCON_FUNC Widget::get_toplevel( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* gwdt = gtk_widget_get_toplevel( (GtkWidget*)_obj );
    Item* wki = vm->findWKI( "GtkWidget" );
    vm->retval( new Widget( wki->asClass(), gwdt ) );
}


//FALCON_FUNC Widget::get_ancestor( VMARG );

//FALCON_FUNC Widget::get_colormap( VMARG );

//FALCON_FUNC Widget::set_colormap( VMARG );

//FALCON_FUNC Widget::get_visual( VMARG );


/*#
    @method get_events GtkWidget
    @brief Returns the event mask for the widget.
    @return (integer)
    (A bitfield containing flags from the GdkEventMask enumeration.)
    These are the events that the widget will receive.
 */
FALCON_FUNC Widget::get_events( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_widget_get_events( (GtkWidget*)_obj ) );
}


//FALCON_FUNC Widget::get_pointer( VMARG );



/*#
    @method is_ancestor GtkWidget
    @brief Determines whether widget is somewhere inside ancestor, possibly with intermediate containers.
    @return (boolean)
 */
FALCON_FUNC Widget::is_ancestor( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !IS_DERIVED( i_wdt, GtkWidget ) )
    {
        throw_inv_params( "GtkWidget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*)((GData*)i_wdt->asObject()->getUserData())->obj();
    vm->retval( (bool) gtk_widget_is_ancestor( (GtkWidget*)_obj, wdt ) );
}


//FALCON_FUNC Widget::translate_coordinates( VMARG );



/*#
    @method hide_on_delete GtkWidget
    @brief Utility function.
    @return (boolean) always true
    Intended to be connected to the "delete-event" signal on a GtkWindow.
    The function calls gtk_widget_hide() on its argument, then returns TRUE.
    If connected to ::delete-event, the result is that clicking the close button
    for a window (on the window frame, top right corner usually) will hide but
    not destroy the window.
    By default, GTK+ destroys windows when ::delete-event is received.
 */
FALCON_FUNC Widget::hide_on_delete( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gboolean b = gtk_widget_hide_on_delete( (GtkWidget*)_obj );
    vm->retval( (bool) b );
}


//FALCON_FUNC Widget::set_style( VMARG );

//FALCON_FUNC Widget::set_rc_style( VMARG );

//FALCON_FUNC Widget::ensure_style( VMARG );

//FALCON_FUNC Widget::get_style( VMARG );

//FALCON_FUNC Widget::restore_default_style( VMARG );

//FALCON_FUNC Widget::reset_rc_styles( VMARG );

//FALCON_FUNC Widget::push_colormap( VMARG );

//FALCON_FUNC Widget::pop_colormap( VMARG );

//FALCON_FUNC Widget::set_default_colormap( VMARG );

//FALCON_FUNC Widget::get_default_style( VMARG );

//FALCON_FUNC Widget::get_default_colormap( VMARG );

//FALCON_FUNC Widget::get_default_visual( VMARG );

//FALCON_FUNC Widget::set_direction( VMARG );

//FALCON_FUNC Widget::get_direction( VMARG );

//FALCON_FUNC Widget::set_default_direction( VMARG );

//FALCON_FUNC Widget::get_default_direction( VMARG );

//FALCON_FUNC Widget::shape_combine_mask( VMARG );

//FALCON_FUNC Widget::input_shape_combine_mask( VMARG );

//FALCON_FUNC Widget::path( VMARG );

//FALCON_FUNC Widget::class_path( VMARG );

//FALCON_FUNC Widget::get_composite_name( VMARG );

//FALCON_FUNC Widget::modify_style( VMARG );

//FALCON_FUNC Widget::get_modifier_style( VMARG );

//FALCON_FUNC Widget::modify_fg( VMARG );

//FALCON_FUNC Widget::modify_bg( VMARG );

//FALCON_FUNC Widget::modify_text( VMARG );

//FALCON_FUNC Widget::modify_base( VMARG );

//FALCON_FUNC Widget::modify_font( VMARG );

//FALCON_FUNC Widget::modify_cursor( VMARG );

//FALCON_FUNC Widget::create_pango_context( VMARG );

//FALCON_FUNC Widget::get_pango_context( VMARG );

//FALCON_FUNC Widget::create_pango_layout( VMARG );

//FALCON_FUNC Widget::widget_render_icon( VMARG );

//FALCON_FUNC Widget::pop_composite_child( VMARG );

//FALCON_FUNC Widget::push_composite_child( VMARG );

//FALCON_FUNC Widget::queue_clear( VMARG );

//FALCON_FUNC Widget::queue_clear_area( VMARG );

//FALCON_FUNC Widget::queue_draw_area( VMARG );

//FALCON_FUNC Widget::reset_shapes( VMARG );

//FALCON_FUNC Widget::set_app_paintable( VMARG );

//FALCON_FUNC Widget::set_double_buffered( VMARG );

//FALCON_FUNC Widget::set_redraw_on_allocate( VMARG );

//FALCON_FUNC Widget::set_composite_name( VMARG );

//FALCON_FUNC Widget::set_scroll_adjustments( VMARG );

//FALCON_FUNC Widget::mnemonic_activate( VMARG );

//FALCON_FUNC Widget::class_install_style_property( VMARG );

//FALCON_FUNC Widget::class_install_style_property_parser( VMARG );

//FALCON_FUNC Widget::class_find_style_property( VMARG );

//FALCON_FUNC Widget::class_list_style_properties( VMARG );

//FALCON_FUNC Widget::region_intersect( VMARG );

//FALCON_FUNC Widget::send_expose( VMARG );

//FALCON_FUNC Widget::style_get( VMARG );

//FALCON_FUNC Widget::style_get_property( VMARG );

//FALCON_FUNC Widget::style_get_valist( VMARG );

//FALCON_FUNC Widget::style_attach( VMARG );

//FALCON_FUNC Widget::get_accessible( VMARG );

//FALCON_FUNC Widget::child_focus( VMARG );

//FALCON_FUNC Widget::child_notify( VMARG );

//FALCON_FUNC Widget::freeze_child_notify( VMARG );

//FALCON_FUNC Widget::get_child_visible( VMARG );

//FALCON_FUNC Widget::get_parent( VMARG );

//FALCON_FUNC Widget::get_settings( VMARG );

//FALCON_FUNC Widget::get_clipboard( VMARG );

//FALCON_FUNC Widget::get_display( VMARG );

//FALCON_FUNC Widget::get_root_window( VMARG );

//FALCON_FUNC Widget::get_screen( VMARG );

//FALCON_FUNC Widget::has_screen( VMARG );


/*#
    @method get_size_request GtkWidget
    @brief Get the size requested for the widget
    @return [ width, height ]

    Gets the size request that was explicitly set for the widget using
    gtk_widget_set_size_request(). A value of -1 stored in width or height
    indicates that that dimension has not been set explicitly and the natural
    requisition of the widget will be used intead. See gtk_widget_set_size_request().

    To get the size a widget will actually use, call gtk_widget_size_request()
    instead of this function.
 */
FALCON_FUNC Widget::get_size_request( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gint w, h;
    gtk_widget_get_size_request( (GtkWidget*)_obj, &w, &h );
    CoreArray* arr = new CoreArray;
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


//FALCON_FUNC Widget::set_child_visible( VMARG );

//FALCON_FUNC Widget::set_default_visual( VMARG );


/*#
    @method set_size_request GtkWidget
    @brief Sets the minimum size of a widget
    @param width (integer)
    @param height (integer)

    that is, the widget's size request will be width by height.
    You can use this function to force a widget to be either larger or smaller
    than it normally would be.

    In most cases, gtk_window_set_default_size() is a better choice for toplevel
    windows than this function; setting the default size will still allow users
    to shrink the window. Setting the size request will force them to leave the
    window at least as large as the size request. When dealing with window sizes,
    gtk_window_set_geometry_hints() can be a useful function as well.

    Note the inherent danger of setting any fixed size - themes, translations
    into other languages, different fonts, and user action can all change the
    appropriate size for a given widget. So, it's basically impossible to hardcode
    a size that will always be correct.

    The size request of a widget is the smallest size a widget can accept while
    still functioning well and drawing itself correctly. However in some strange
    cases a widget may be allocated less than its requested size, and in many
    cases a widget may be allocated more space than it requested.

    If the size request in a given direction is -1 (unset), then the "natural"
    size request of the widget will be used instead.

    Widgets can't actually be allocated a size less than 1 by 1, but you can pass
    0,0 to this function to mean "as small as possible."
 */
FALCON_FUNC Widget::set_size_request( VMARG )
{
    Item* i_w = vm->param( 0 );
    Item* i_h = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || i_w->isNil() || !i_w->isInteger()
        || !i_h || i_h->isNil() || !i_h->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_widget_set_size_request( (GtkWidget*)_obj, i_w->asInteger(), i_h->asInteger() );
}


//FALCON_FUNC Widget::set_visual( VMARG );

//FALCON_FUNC Widget::thaw_child_notify( VMARG );

//FALCON_FUNC Widget::set_no_show_all( VMARG );

//FALCON_FUNC Widget::get_no_show_all( VMARG );

//FALCON_FUNC Widget::list_mnemonic_labels( VMARG );

//FALCON_FUNC Widget::add_mnemonic_label( VMARG );

//FALCON_FUNC Widget::remove_mnemonic_label( VMARG );

//FALCON_FUNC Widget::get_action( VMARG );

//FALCON_FUNC Widget::is_composited( VMARG );

//FALCON_FUNC Widget::error_bell( VMARG );

//FALCON_FUNC Widget::keynav_failed( VMARG );

//FALCON_FUNC Widget::get_tooltip_markup( VMARG );

//FALCON_FUNC Widget::set_tooltip_markup( VMARG );

//FALCON_FUNC Widget::get_tooltip_text( VMARG );

//FALCON_FUNC Widget::set_tooltip_text( VMARG );

//FALCON_FUNC Widget::get_tooltip_window( VMARG );

//FALCON_FUNC Widget::tooltip_window( VMARG );

//FALCON_FUNC Widget::get_has_tooltip( VMARG );

//FALCON_FUNC Widget::set_has_tooltip( VMARG );

//FALCON_FUNC Widget::trigger_tooltip_query( VMARG );

//FALCON_FUNC Widget::get_snapshot( VMARG );

//FALCON_FUNC Widget::get_window( VMARG );

//FALCON_FUNC Widget::get_allocation( VMARG );

//FALCON_FUNC Widget::set_allocation( VMARG );

//FALCON_FUNC Widget::get_app_paintable( VMARG );

//FALCON_FUNC Widget::get_can_default( VMARG );

//FALCON_FUNC Widget::set_can_default( VMARG );

//FALCON_FUNC Widget::get_can_focus( VMARG );

//FALCON_FUNC Widget::set_can_focus( VMARG );

//FALCON_FUNC Widget::get_double_buffered( VMARG );

//FALCON_FUNC Widget::get_has_window( VMARG );

//FALCON_FUNC Widget::set_has_window( VMARG );

//FALCON_FUNC Widget::get_sensitive( VMARG );

//FALCON_FUNC Widget::is_sensitive( VMARG );

//FALCON_FUNC Widget::get_state( VMARG );

//FALCON_FUNC Widget::get_visible( VMARG );

//FALCON_FUNC Widget::set_visible( VMARG );

//FALCON_FUNC Widget::has_default( VMARG );

//FALCON_FUNC Widget::has_focus( VMARG );

//FALCON_FUNC Widget::has_grab( VMARG );

//FALCON_FUNC Widget::has_rc_style( VMARG );

//FALCON_FUNC Widget::is_drawable( VMARG );

//FALCON_FUNC Widget::is_toplevel( VMARG );

//FALCON_FUNC Widget::set_window( VMARG );

//FALCON_FUNC Widget::set_receives_default( VMARG );

//FALCON_FUNC Widget::get_receives_default( VMARG );

//FALCON_FUNC Widget::set_realized( VMARG );

//FALCON_FUNC Widget::get_realized( VMARG );

//FALCON_FUNC Widget::set_mapped( VMARG );

//FALCON_FUNC Widget::get_mapped( VMARG );

//FALCON_FUNC Widget::get_requisition( VMARG );


} // Gtk
} // Falcon
