import asyncio

async def task1():
    print("Start task1")
    await asyncio.sleep(2)
    print("End task1")

async def task2():
    print("Start task2")
    await asyncio.sleep(1)
    print("End task2")

async def chain():
    t1 = asyncio.create_task(task1())
    t2 = asyncio.create_task(task2())
    await t1
    await t2

if __name__ == "__main__":
    asyncio.run(chain())
