use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use flecs_ecs::prelude::QueryAPI;

const LOOPS: usize = 100_000;
const FRAGMENTED_ENTITIES_PER_TYPE: usize = 20;
const HEAVY_COMPUTE_ITERATIONS: usize = 100;

use bevy_ecs::prelude::{Component as BevyComponent, World as BevyWorld};

use flecs_ecs::prelude::Component as FlecsComponent;
use flecs_ecs::prelude::World as FlecsWorld;
use hecs::World as HecsWorld;
use legion::{world::World as LegionWorld, IntoQuery};
use nalgebra::Matrix4;
use specs::{
    Builder, Component as SpecsComponent, Join, VecStorage, World as SpecsWorld, WorldExt,
};

#[derive(Debug, Clone, Copy, Default, BevyComponent, FlecsComponent)]
struct Position {
    x: f32,
    y: f32,
}
#[derive(Debug, Clone, Copy, Default, BevyComponent, FlecsComponent)]
struct Velocity {
    x: f32,
    y: f32,
}
#[derive(Debug, Clone, Copy, Default, BevyComponent, FlecsComponent)]
struct Transform(pub Matrix4<f32>);
#[derive(Debug, Clone, Copy, Default, BevyComponent, FlecsComponent)]
struct Data(f32);

impl SpecsComponent for Position {
    type Storage = VecStorage<Self>;
}
impl SpecsComponent for Velocity {
    type Storage = VecStorage<Self>;
}
impl SpecsComponent for Transform {
    type Storage = VecStorage<Self>;
}
impl SpecsComponent for Data {
    type Storage = VecStorage<Self>;
}

macro_rules! define_fragmented {
    ($($name:ident),*) => {
        $(#[derive(Debug, Clone, Copy, Default, BevyComponent,FlecsComponent)]
        struct $name(f32);
        impl SpecsComponent for $name { type Storage = VecStorage<Self>; })*
    }
}
define_fragmented!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);

fn bench_spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn");

    group.bench_function(BenchmarkId::new("bevy", LOOPS), |b| {
        let mut world = BevyWorld::default();

        b.iter(|| {
            world.clear_all();

            for i in 0..LOOPS {
                world.spawn((
                    Position {
                        x: i as f32,
                        y: i as f32,
                    },
                    Velocity {
                        x: i as f32,
                        y: i as f32,
                    },
                ));
            }
        });
    });

    group.bench_function(BenchmarkId::new("hecs", LOOPS), |b| {
        let mut world = HecsWorld::new();

        b.iter(|| {
            world.clear();

            for i in 0..LOOPS {
                world.spawn((
                    Position {
                        x: i as f32,
                        y: i as f32,
                    },
                    Velocity {
                        x: i as f32,
                        y: i as f32,
                    },
                ));
            }
        });
    });

    group.bench_function(BenchmarkId::new("flecs", LOOPS), |b| {
        let world = FlecsWorld::new();
        world.component::<Position>();
        world.component::<Velocity>();

        b.iter(|| {
            world.remove_all::<Position>();
            world.remove_all::<Velocity>();

            for i in 0..LOOPS {
                world
                    .entity()
                    .set(Position {
                        x: i as f32,
                        y: i as f32,
                    })
                    .set(Velocity {
                        x: i as f32,
                        y: i as f32,
                    });
            }
        });
    });

    group.bench_function(BenchmarkId::new("specs", LOOPS), |b| {
        let mut world = SpecsWorld::new();
        world.register::<Position>();
        world.register::<Velocity>();

        b.iter(|| {
            world.delete_all();

            for i in 0..LOOPS {
                world
                    .create_entity()
                    .with(Position {
                        x: i as f32,
                        y: i as f32,
                    })
                    .with(Velocity {
                        x: i as f32,
                        y: i as f32,
                    })
                    .build();
            }
        });
    });

    group.bench_function(BenchmarkId::new("legion", LOOPS), |b| {
        let mut world = LegionWorld::default();

        b.iter(|| {
            world.clear();

            for i in 0..LOOPS {
                world.push((
                    Position {
                        x: i as f32,
                        y: i as f32,
                    },
                    Velocity {
                        x: i as f32,
                        y: i as f32,
                    },
                ));
            }
        });
    });

    group.finish();
}

fn bench_simple_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_iter");

    group.bench_function("bevy", |b| {
        let mut world = BevyWorld::default();

        b.iter(|| {
            world.clear_all();

            for _ in 0..LOOPS {
                world.spawn((Position::default(), Velocity::default()));
            }

            for mut q in world
                .query::<(&mut Position, &Velocity)>()
                .iter_mut(&mut world)
            {
                q.0.x += q.1.x;
                q.0.y += q.1.y;
            }
        });
    });

    group.bench_function("hecs", |b| {
        let mut world = HecsWorld::new();

        b.iter(|| {
            world.clear();

            for _ in 0..LOOPS {
                world.spawn((Position::default(), Velocity::default()));
            }

            for (_entity, (pos, vel)) in world.query_mut::<(&mut Position, &Velocity)>() {
                pos.x += vel.x;
                pos.y += vel.y;
            }
        });
    });

    group.bench_function("flecs", |b| {
        let world = FlecsWorld::new();
        world.component::<Position>();
        world.component::<Velocity>();

        b.iter(|| {
            world.remove_all::<Position>();
            world.remove_all::<Velocity>();

            let mut ents = Vec::with_capacity(LOOPS);
            for _ in 0..LOOPS {
                ents.push(
                    world
                        .entity()
                        .set(Position::default())
                        .set(Velocity::default()),
                );
            }

            for e in ents.iter_mut() {
                e.get::<(&mut Position, &Velocity)>(|(p, v)| {
                    p.x += v.x;
                    p.y += v.y;
                });
            }
        });
    });

    group.bench_function("specs", |b| {
        let mut world = SpecsWorld::new();
        world.register::<Position>();
        world.register::<Velocity>();

        b.iter(|| {
            world.delete_all();

            for _ in 0..LOOPS {
                world
                    .create_entity()
                    .with(Position::default())
                    .with(Velocity::default())
                    .build();
            }
            let mut ps = world.write_storage::<Position>();
            let vs = world.read_storage::<Velocity>();
            for (p, v) in (&mut ps, &vs).join() {
                p.x += v.x;
                p.y += v.y;
            }
        });
    });

    group.bench_function("legion", |b| {
        let mut world = LegionWorld::default();

        b.iter(|| {
            world.clear();

            for _ in 0..LOOPS {
                world.push((Position::default(), Velocity::default()));
            }
            for (p, v) in <(&mut Position, &Velocity)>::query().iter_mut(&mut world) {
                p.x += v.x;
                p.y += v.y;
            }
        });
    });

    group.finish();
}

fn bench_fragmented_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmented_iter");

    group.bench_function("hecs", |b| {
        let mut world = HecsWorld::new();

        b.iter(|| {
            world.clear();

            world.spawn((A(0.0), Data(1.0)));
            world.spawn((B(0.0), Data(1.0)));
            world.spawn((C(0.0), Data(1.0)));

            for _ in 0..FRAGMENTED_ENTITIES_PER_TYPE {
                world.spawn((A(0.0), Data(1.0)));
                world.spawn((B(0.0), Data(1.0)));
                world.spawn((C(0.0), Data(1.0)));
            }

            for (_entity, data) in world.query_mut::<&mut Data>() {
                data.0 *= 2.0;
            }
        });
    });

    group.bench_function("flecs", |b| {
        let world = FlecsWorld::new();
        world.component::<Data>();
        world.component::<A>();
        world.component::<B>();
        world.component::<C>();

        b.iter(|| {
            world.remove_all::<Data>();
            world.remove_all::<A>();
            world.remove_all::<B>();
            world.remove_all::<C>();

            for _ in 0..FRAGMENTED_ENTITIES_PER_TYPE {
                world.entity().set(A(0.0)).set(Data(1.0));
                world.entity().set(B(0.0)).set(Data(1.0));
                world.entity().set(C(0.0)).set(Data(1.0));
            }

            let query = world.new_query::<&mut Data>();
            query.each_iter(|_, _, d| {
                d.0 *= 2.0;
            });
        });
    });

    group.bench_function("specs", |b| {
        let mut world = SpecsWorld::new();
        world.register::<Data>();
        world.register::<A>();
        world.register::<B>();
        world.register::<C>();

        b.iter(|| {
            world.delete_all();

            for _ in 0..FRAGMENTED_ENTITIES_PER_TYPE {
                world.create_entity().with(A(0.0)).with(Data(1.0)).build();
                world.create_entity().with(B(0.0)).with(Data(1.0)).build();
                world.create_entity().with(C(0.0)).with(Data(1.0)).build();
            }

            let mut ds = world.write_storage::<Data>();
            for d in (&mut ds).join() {
                d.0 *= 2.0;
            }
        });
    });

    group.bench_function("legion", |b| {
        let mut world = LegionWorld::default();

        b.iter(|| {
            world.clear();

            for _ in 0..FRAGMENTED_ENTITIES_PER_TYPE {
                world.push((A(0.0), Data(1.0)));
                world.push((B(0.0), Data(1.0)));
                world.push((C(0.0), Data(1.0)));
            }

            for d in <&mut Data>::query().iter_mut(&mut world) {
                d.0 *= 2.0;
            }
        });
    });

    group.bench_function("bevy", |b| {
        let mut world = BevyWorld::default();

        b.iter(|| {
            world.clear_all();

            for _ in 0..FRAGMENTED_ENTITIES_PER_TYPE {
                world.spawn((A(0.0), Data(1.0)));
                world.spawn((B(0.0), Data(1.0)));
                world.spawn((C(0.0), Data(1.0)));
            }

            for mut q in world.query::<&mut Data>().iter_mut(&mut world) {
                q.0 *= 2.0;
            }
        });
    });

    group.finish();
}

fn bench_heavy_compute(c: &mut Criterion) {
    let identity = Matrix4::identity();
    let mut group = c.benchmark_group("heavy_compute");

    group.bench_function("hecs", |b| {
        let mut world = HecsWorld::new();

        b.iter(|| {
            world.clear();

            for _ in 0..1000 {
                world.spawn((Transform(Matrix4::identity()),));
            }

            for (_entity, transform) in world.query_mut::<&mut Transform>() {
                for _ in 0..HEAVY_COMPUTE_ITERATIONS {
                    transform.0 = transform.0 * identity;
                }
            }
        });
    });

    group.bench_function("flecs", |b| {
        let world = FlecsWorld::new();
        world.component::<Transform>();

        b.iter(|| {
            world.remove_all::<Transform>();

            let mut ents = Vec::with_capacity(1000);
            for _ in 0..1000 {
                ents.push(world.entity().set(Transform(Matrix4::identity())));
            }
            for e in ents.iter_mut() {
                e.get::<&mut Transform>(|t| {
                    for _ in 0..HEAVY_COMPUTE_ITERATIONS {
                        t.0 = t.0 * identity;
                    }
                });
            }
        });
    });

    group.bench_function("specs", |b| {
        let mut world = SpecsWorld::new();

        b.iter(|| {
            world.delete_all();

            world.register::<Transform>();
            for _ in 0..1000 {
                world
                    .create_entity()
                    .with(Transform(Matrix4::identity()))
                    .build();
            }
            let mut ts = world.write_storage::<Transform>();
            for t in (&mut ts).join() {
                for _ in 0..HEAVY_COMPUTE_ITERATIONS {
                    t.0 = t.0 * identity;
                }
            }
        });
    });

    group.bench_function("legion", |b| {
        let mut world = LegionWorld::default();

        b.iter(|| {
            world.clear();

            for _ in 0..1000 {
                world.push((Transform(Matrix4::identity()),));
            }
            for t in <&mut Transform>::query().iter_mut(&mut world) {
                for _ in 0..HEAVY_COMPUTE_ITERATIONS {
                    t.0 = t.0 * identity;
                }
            }
        });
    });

    group.bench_function("bevy", |b| {
        let mut world = BevyWorld::default();

        b.iter(|| {
            world.clear_all();

            for _ in 0..1000 {
                world.spawn((Transform(Matrix4::identity()),));
            }
            for mut q in world.query::<&mut Transform>().iter_mut(&mut world) {
                for _ in 0..HEAVY_COMPUTE_ITERATIONS {
                    q.0 = q.0 * identity;
                }
            }
        });
    });

    group.finish();
}

fn bench_crud_add_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("crud_add_remove");

    group.bench_function("hecs", |b| {
        let mut world = HecsWorld::new();

        b.iter(|| {
            world.clear();

            let mut ents = Vec::with_capacity(LOOPS);

            for _ in 0..LOOPS {
                ents.push(world.spawn((A(0.0),)));
            }

            for &e in &ents {
                world.insert_one(e, B(0.0)).unwrap();
            }

            for &e in &ents {
                world.remove_one::<B>(e).unwrap();
            }
        });
    });

    group.bench_function("specs", |b| {
        let mut world = SpecsWorld::new();
        world.register::<A>();
        world.register::<B>();

        b.iter(|| {
            world.delete_all();

            let mut ents = Vec::with_capacity(LOOPS);

            for _ in 0..LOOPS {
                ents.push(world.create_entity().with(A(0.0)).build());
            }

            {
                let mut storage = world.write_storage::<B>();
                for &e in &ents {
                    storage.insert(e, B(0.0)).unwrap();
                }
            }

            {
                let mut storage = world.write_storage::<B>();
                for &e in &ents {
                    storage.remove(e);
                }
            }
        });
    });

    group.bench_function("flecs", |b| {
        let world = FlecsWorld::new();
        world.component::<A>();
        world.component::<B>();

        b.iter(|| {
            world.remove_all::<A>();
            world.remove_all::<B>();

            let mut ents = Vec::with_capacity(LOOPS);

            for _ in 0..LOOPS {
                ents.push(world.entity().set(A(0.0)));
            }

            for e in &ents {
                e.set(B(0.0));
            }

            for e in &ents {
                e.remove::<B>();
            }
        });
    });

    group.bench_function("legion", |b| {
        let mut world = LegionWorld::default();

        b.iter(|| {
            world.clear();

            let ents: Vec<_> = (0..LOOPS).map(|_| world.push((A(0.0),))).collect();

            for &e in &ents {
                let mut entry = world.entry(e).unwrap();
                entry.add_component(B(0.0));
            }

            for &e in &ents {
                let mut entry = world.entry(e).unwrap();
                entry.remove_component::<B>();
            }
        });
    });

    group.bench_function("bevy", |b| {
        let mut world = BevyWorld::default();

        b.iter(|| {
            world.clear_all();

            let mut ids = Vec::with_capacity(LOOPS);

            for _ in 0..LOOPS {
                ids.push(world.spawn((A(0.0),)).id());
            }

            for &id in &ids {
                world.entity_mut(id).insert(B(0.0));
            }

            for &id in &ids {
                world.entity_mut(id).remove::<B>();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_spawn,
    bench_simple_iter,
    bench_fragmented_iter,
    bench_heavy_compute,
    bench_crud_add_remove,
);
criterion_main!(benches);
